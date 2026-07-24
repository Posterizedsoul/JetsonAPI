"""Public API. The envelope is stable across model swaps: what changes between
models is the CONTENT of `results`, never the shape of the response."""

import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import (
    APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
)

from app import auth, config, db, storage
from app.runner import ManifestError, read_metadata, runner

router = APIRouter(prefix="/v1")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _log_error(kind: str, detail: str, context: dict | None = None) -> None:
    db.execute(
        "INSERT INTO errors (kind, detail, context) VALUES (%s,%s,%s)",
        (kind, detail[:2000], json.dumps(context or {})),
    )


# ------------------------------------------------------------------- models --

@router.post("/models")
async def register_model(
    archive: UploadFile = File(...),
    model_id: str = Form(...),
    version: str = Form(...),
    activate: bool = Form(False),
    _: dict = Depends(auth.require_admin),
) -> dict:
    """Register a TorchScript archive. The embedded metadata.json IS the
    manifest — task, classes, input size and preprocessing all come from it.
    Adding a model is this call plus nothing else."""
    data = await archive.read()
    dest = Path(config.MODEL_DIR) / f"{model_id}__{version}.ts.pt"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)

    try:
        meta = read_metadata(dest)
    except ManifestError as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(422, f"rejected: {exc}") from exc

    row_id = uuid.uuid4()
    try:
        db.execute(
            "INSERT INTO models (id, model_id, version, task, archive_key, meta, classes)"
            " VALUES (%s,%s,%s,%s,%s,%s,%s)",
            (row_id, model_id, version, meta["task"], str(dest),
             json.dumps(meta), json.dumps(meta["classes"])),
        )
    except Exception as exc:
        if "unique" in str(exc).lower():
            raise HTTPException(409, f"{model_id}:{version} already registered") from exc
        raise

    if activate:
        _activate(str(row_id))
    return {
        "id": str(row_id), "model_id": model_id, "version": version,
        "task": meta["task"], "classes": meta["classes"],
        "val_metrics": meta.get("val_metrics", {}), "active": activate,
    }


@router.get("/models")
def list_models(_: dict = Depends(auth.require_key)) -> dict:
    rows = db.query(
        "SELECT id, model_id, version, task, classes, active, registered_at,"
        " meta->'val_metrics' AS val_metrics, meta->>'variant' AS variant"
        " FROM models ORDER BY registered_at DESC"
    )
    for r in rows:
        r["id"] = str(r["id"])
        r["loaded"] = runner.is_loaded(r["id"])
    return {"models": rows}


def _activate(model_uuid: str) -> dict:
    rows = db.query("SELECT task FROM models WHERE id = %s", (model_uuid,))
    if not rows:
        raise HTTPException(404, "no such model")
    task = rows[0]["task"]
    with db.pool.connection() as conn:
        # One active model per task; the partial unique index enforces it, so
        # deactivate the incumbent in the same transaction.
        conn.execute("UPDATE models SET active = FALSE WHERE task = %s AND active", (task,))
        conn.execute("UPDATE models SET active = TRUE WHERE id = %s", (model_uuid,))
    return {"activated": model_uuid, "task": task}


@router.post("/models/{model_uuid}/activate")
def activate_model(model_uuid: str, _: dict = Depends(auth.require_admin)) -> dict:
    return _activate(model_uuid)


@router.post("/models/{model_uuid}/load")
def load_model(model_uuid: str, _: dict = Depends(auth.require_admin)) -> dict:
    rows = db.query("SELECT archive_key FROM models WHERE id = %s", (model_uuid,))
    if not rows:
        raise HTTPException(404, "no such model")
    meta = runner.load(rows[0]["archive_key"], model_uuid)
    return {"loaded": model_uuid, "task": meta["task"], "classes": meta["classes"]}


@router.post("/models/{model_uuid}/unload")
def unload_model(model_uuid: str, _: dict = Depends(auth.require_admin)) -> dict:
    if runner.is_loaded(model_uuid):
        runner.unload()
    return {"unloaded": model_uuid}


def active_model(task: str) -> dict | None:
    rows = db.query(
        "SELECT id, model_id, version, task, archive_key, classes"
        " FROM models WHERE task = %s AND active",
        (task,),
    )
    return rows[0] if rows else None


# ------------------------------------------------------------------- ingest --

@router.post("/boards")
async def ingest_board(
    background: BackgroundTasks,
    board_id: str = Form(...),
    views: list[UploadFile] = File(...),
    lighting: list[str] = Form(...),
    captured_at: str | None = Form(None),
    task: str = Form("classification"),
    edge_prediction: str | None = Form(None),
    meta: str | None = Form(None),
    run_inference: bool = Form(True),
    key: dict = Depends(auth.require_ingest),
) -> dict:
    """Upload one board with 1..N views.

    Idempotent on board_id + per-view image hash: a client draining a queued
    upload after a failed attempt gets the existing record back rather than a
    duplicate. Originals are stored byte-for-byte; previews are separate.
    """
    if len(views) != len(lighting):
        raise HTTPException(422, "one lighting label per view is required")

    device_id = key["device_id"]
    meta_obj = json.loads(meta) if meta else {}
    edge = json.loads(edge_prediction) if edge_prediction else None

    existing = db.query("SELECT id FROM boards WHERE board_id = %s", (board_id,))
    if existing:
        board_uuid = existing[0]["id"]
        created = False
    else:
        board_uuid = uuid.uuid4()
        db.execute(
            "INSERT INTO boards (id, board_id, device_id, captured_at, meta)"
            " VALUES (%s,%s,%s,%s,%s) ON CONFLICT (board_id) DO NOTHING",
            (board_uuid, board_id, device_id, captured_at, json.dumps(meta_obj)),
        )
        rows = db.query("SELECT id FROM boards WHERE board_id = %s", (board_id,))
        board_uuid, created = rows[0]["id"], True

    stored = []
    for upload, light in zip(views, lighting):
        raw = await upload.read()
        digest = hashlib.sha256(raw).hexdigest()
        dupe = db.query(
            "SELECT id FROM views WHERE board = %s AND image_hash = %s",
            (board_uuid, digest),
        )
        if dupe:
            stored.append({"lighting": light, "image_hash": digest, "duplicate": True})
            continue

        key_orig = f"boards/{board_id}/{digest}{Path(upload.filename or '').suffix or '.bin'}"
        # The original goes in untouched: no re-encode, no colour correction.
        storage.put(key_orig, raw, upload.content_type or "application/octet-stream")
        preview, width, height, fmt = storage.make_preview(raw)
        key_prev = f"previews/{board_id}/{digest}.jpg"
        storage.put(key_prev, preview, "image/jpeg")

        db.execute(
            "INSERT INTO views (id, board, lighting, image_hash, object_key,"
            " preview_key, width, height, format, byte_size)"
            " VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (uuid.uuid4(), board_uuid, light, digest, key_orig, key_prev,
             width, height, fmt, len(raw)),
        )
        stored.append({"lighting": light, "image_hash": digest, "duplicate": False,
                       "width": width, "height": height, "format": fmt})

    if device_id:
        db.execute(
            "UPDATE devices SET last_seen_at = now(),"
            " app_version = COALESCE(%s, app_version),"
            " edge_model_id = COALESCE(%s, edge_model_id),"
            " edge_model_version = COALESCE(%s, edge_model_version)"
            " WHERE device_id = %s",
            (meta_obj.get("app_version"),
             (edge or {}).get("model_id"), (edge or {}).get("model_version"),
             device_id),
        )

    if edge and created:
        _store_edge_prediction(board_uuid, task, edge)

    queued = False
    if run_inference and any(not s["duplicate"] for s in stored):
        background.add_task(run_server_inference, str(board_uuid), task)
        queued = True

    return {
        "board_id": board_id,
        "id": str(board_uuid),
        "created": created,
        "views": stored,
        "inference_queued": queued,
    }


def _store_edge_prediction(board_uuid, task: str, edge: dict) -> None:
    probs = edge.get("probs") or {}
    label = edge.get("label")
    ordered = sorted(probs.values(), reverse=True)
    db.execute(
        "INSERT INTO predictions (id, board, source, task, edge_model_id,"
        " edge_model_ver, probs, label, confidence, margin, view_attention)"
        " VALUES (%s,%s,'edge',%s,%s,%s,%s,%s,%s,%s,%s)",
        (uuid.uuid4(), board_uuid, task, edge.get("model_id"),
         edge.get("model_version"), json.dumps(probs), label,
         edge.get("confidence") or (probs.get(label) if label else None),
         (ordered[0] - ordered[1]) if len(ordered) > 1 else None,
         json.dumps(edge.get("view_attention")) if edge.get("view_attention") else None),
    )


# ---------------------------------------------------------------- inference --

def run_server_inference(board_uuid: str, task: str, replay_job: str | None = None,
                         model_row: dict | None = None) -> str | None:
    """Run the active (or given) model over a stored board and append a row.

    Never updates or deletes: replay and ingest both append, so a model's
    history against real field data stays intact.
    """
    model = model_row or active_model(task)
    if not model:
        _log_error("inference", f"no active model for task {task}", {"board": board_uuid})
        return None
    try:
        if not runner.is_loaded(str(model["id"])):
            runner.load(model["archive_key"], str(model["id"]))

        views = db.query(
            "SELECT object_key FROM views WHERE board = %s ORDER BY created_at, id",
            (board_uuid,),
        )
        if not views:
            return None
        max_views = runner.meta["max_views"]
        payload = [storage.get(v["object_key"]) for v in views[:max_views]]
        out = runner.predict(payload)

        agrees, ordinal_error = _compare_to_edge(board_uuid, task, out, model["classes"])
        pred_id = uuid.uuid4()
        db.execute(
            "INSERT INTO predictions (id, board, source, model, task, probs, label,"
            " confidence, margin, view_attention, outputs, agrees, ordinal_error,"
            " latency_ms, tta, replay_job)"
            " VALUES (%s,%s,'server',%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (pred_id, board_uuid, model["id"], task,
             json.dumps(out.get("probs") or {}), out.get("label"),
             out.get("confidence"), out.get("margin"),
             json.dumps(out.get("view_attention")),
             json.dumps({k: v for k, v in out.items()
                         if k in ("detections", "coverage", "patch_attention",
                                  "mask_shape", "calibrated")}),
             agrees, ordinal_error, out.get("latency_ms"), out.get("tta"), replay_job),
        )
        return str(pred_id)
    except Exception as exc:
        _log_error("inference", str(exc), {"board": board_uuid, "task": task})
        return None


@router.get("/boards")
def list_boards(
    limit: int = 50,
    offset: int = 0,
    device_id: str | None = None,
    task: str | None = None,
    flagged: bool | None = None,
    verified: bool | None = None,
    sort: str = "disagreement",
    _: dict = Depends(auth.require_key),
) -> dict:
    """Review queue backing. Default sort is what review actually needs:
    biggest ordinal error first, then narrowest top-two margin."""
    order = {
        # NULLS LAST so boards without a comparison sink below real conflicts.
        "disagreement": "p.ordinal_error DESC NULLS LAST, p.margin ASC NULLS LAST",
        "margin": "p.margin ASC NULLS LAST",
        "confidence": "p.confidence ASC NULLS LAST",
        "recent": "b.received_at DESC",
    }.get(sort)
    if order is None:
        raise HTTPException(422, f"unknown sort {sort!r}")

    where, params = ["TRUE"], []
    if device_id:
        where.append("b.device_id = %s"); params.append(device_id)
    if task:
        where.append("p.task = %s"); params.append(task)
    if flagged is not None:
        where.append("b.flagged = %s"); params.append(flagged)
    if verified is not None:
        where.append(("EXISTS" if verified else "NOT EXISTS") +
                     " (SELECT 1 FROM verifications v WHERE v.board = b.id)")

    rows = db.query(
        "SELECT b.id, b.board_id, b.device_id, b.captured_at, b.received_at,"
        " b.flagged, p.label AS server_label, p.confidence, p.margin,"
        " p.ordinal_error, p.agrees, e.label AS edge_label,"
        " (SELECT count(*) FROM views v WHERE v.board = b.id) AS view_count,"
        " (SELECT label FROM verifications x WHERE x.board = b.id"
        "  ORDER BY created_at DESC LIMIT 1) AS verified_label"
        " FROM boards b"
        " LEFT JOIN LATERAL (SELECT * FROM predictions WHERE board = b.id"
        "   AND source = 'server' ORDER BY created_at DESC LIMIT 1) p ON TRUE"
        " LEFT JOIN LATERAL (SELECT * FROM predictions WHERE board = b.id"
        "   AND source = 'edge' ORDER BY created_at DESC LIMIT 1) e ON TRUE"
        f" WHERE {' AND '.join(where)} ORDER BY {order} LIMIT %s OFFSET %s",
        tuple(params) + (limit, offset),
    )
    for r in rows:
        r["id"] = str(r["id"])
    return {"boards": rows, "limit": limit, "offset": offset}


@router.get("/boards/{board_id}")
def get_board(board_id: str, _: dict = Depends(auth.require_key)) -> dict:
    rows = db.query("SELECT * FROM boards WHERE board_id = %s", (board_id,))
    if not rows:
        raise HTTPException(404, "no such board")
    board = rows[0]
    board["id"] = str(board["id"])

    views = db.query(
        "SELECT id, lighting, image_hash, width, height, format, byte_size"
        " FROM views WHERE board = %s ORDER BY created_at, id", (board["id"],)
    )
    preds = db.query(
        "SELECT p.id, p.source, p.task, p.label, p.confidence, p.margin, p.probs,"
        " p.view_attention, p.outputs, p.agrees, p.ordinal_error, p.latency_ms,"
        " p.tta, p.created_at, p.replay_job,"
        " m.model_id, m.version AS model_version, p.edge_model_id, p.edge_model_ver"
        " FROM predictions p LEFT JOIN models m ON m.id = p.model"
        " WHERE p.board = %s ORDER BY p.created_at", (board["id"],)
    )
    verifications = db.query(
        "SELECT label, corrected, note, verified_by, created_at FROM verifications"
        " WHERE board = %s ORDER BY created_at DESC", (board["id"],)
    )
    for v in views:
        v["id"] = str(v["id"])
    for p in preds:
        p["id"] = str(p["id"])
        p["replay_job"] = str(p["replay_job"]) if p["replay_job"] else None
    return {"board": board, "views": views, "predictions": preds,
            "verifications": verifications}


def _compare_to_edge(board_uuid: str, task: str, out: dict,
                     classes: list) -> tuple[bool | None, int | None]:
    """Exact-match agreement plus ordinal distance.

    Ordinal distance is index difference in the model's OWN class list: the
    server knows an ordering exists, not what the grades mean. Confusing
    adjacent grades is a much smaller error than skipping one, and a mean of
    this is the number worth watching.
    """
    rows = db.query(
        "SELECT label FROM predictions WHERE board = %s AND source = 'edge'"
        " AND task = %s ORDER BY created_at DESC LIMIT 1",
        (board_uuid, task),
    )
    if not rows or not rows[0]["label"] or not out.get("label"):
        return None, None
    edge_label = rows[0]["label"]
    if edge_label not in classes or out["label"] not in classes:
        # Edge model has a different class set — agreement is meaningless.
        return None, None
    return (edge_label == out["label"],
            abs(classes.index(edge_label) - classes.index(out["label"])))
