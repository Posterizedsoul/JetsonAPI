"""Server-rendered admin UI. Jinja + HTMX, no build step, no SPA.

Kept deliberately thin: every handler is a DB read or a call into the same
helpers the JSON API uses (routes.save_and_register, routes._activate,
runner.load/unload), then it renders a template. Business logic lives in
routes.py / runner.py, not here.

Model-agnostic: nothing here names a class or a task. Class lists come from
each model's metadata and are rendered as-is.
"""

from pathlib import Path

import qrcode
from fastapi import APIRouter, Depends, File, Form, Header, Request, UploadFile
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from app import auth, config, db, logbuffer, metrics, routes
from app.runner import runner

templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
router = APIRouter(include_in_schema=False)


def current_admin(request: Request) -> dict | None:
    return auth.admin_from_cookie(request.cookies.get(auth.COOKIE_NAME))


def require_web_admin(request: Request) -> dict:
    """Dependency for UI routes. Raises a redirect to the login page rather
    than a 401, so an unauthenticated browser lands somewhere useful."""
    admin = current_admin(request)
    if not admin:
        raise _Redirect("/ui/login")
    return admin


class _Redirect(Exception):
    def __init__(self, url: str) -> None:
        self.url = url


# --------------------------------------------------------------------- login --

@router.get("/ui/login", response_class=HTMLResponse)
def login_page(request: Request, t: str | None = None):
    # Scanned a QR: swap the one-time token for the session cookie.
    if t:
        key = auth.consume_login_token(t)
        if key and auth.admin_from_cookie(key):
            resp = RedirectResponse("/ui", status_code=303)
            resp.set_cookie(auth.COOKIE_NAME, key, httponly=True,
                            samesite="lax", max_age=config.SESSION_DAYS * 86400)
            return resp
        return templates.TemplateResponse(
            request, "login.html",
            {"error": "That login link expired or was already used. Reprint the QR."},
            status_code=401,
        )
    if current_admin(request):
        return RedirectResponse("/ui", status_code=303)
    return templates.TemplateResponse(request, "login.html", {"error": None})


def _qr_ansi(data: str) -> str:
    """A terminal QR that actually scans. Each module is two spaces so it's
    square, coloured with ANSI background so it's real black-on-white whatever
    the terminal theme is (half-block glyphs from print_ascii misalign and
    scanners choke on them). border=2 is the quiet zone."""
    qr = qrcode.QRCode(border=2)
    qr.add_data(data)
    qr.make(fit=True)
    white, black, reset = "\033[107m", "\033[40m", "\033[0m"
    lines = []
    for row in qr.get_matrix():
        cells = "".join((black if cell else white) + "  " for cell in row)
        lines.append(cells + reset)
    return "\n".join(lines)


@router.get("/ui/login-qr", response_class=PlainTextResponse)
def login_qr(base: str, x_api_key: str = Header(...),
             _: dict = Depends(auth.require_admin)):
    """Mint a one-time login link and render it as a terminal QR. Called by the
    setup script's access panel with an admin key; `base` is the reachable
    origin (e.g. http://100.x:8000) since the gateway can't know its own
    Tailscale address."""
    token = auth.mint_login_token(x_api_key)
    url = f"{base.rstrip('/')}/ui/login?t={token}"
    return f"{_qr_ansi(url)}\n\nScan to log in (one use, {auth.LOGIN_TOKEN_TTL // 60} min):\n{url}\n"


@router.post("/ui/login")
def login(request: Request, api_key: str = Form(...)):
    if not auth.admin_from_cookie(api_key):
        return templates.TemplateResponse(
            request, "login.html",
            {"error": "Not a valid admin key."}, status_code=401,
        )
    resp = RedirectResponse("/ui", status_code=303)
    resp.set_cookie(
        auth.COOKIE_NAME, api_key, httponly=True, samesite="lax", max_age=config.SESSION_DAYS * 86400
    )
    return resp


@router.get("/ui/logout")
def logout():
    resp = RedirectResponse("/ui/login", status_code=303)
    resp.delete_cookie(auth.COOKIE_NAME)
    return resp


@router.get("/ui/help", response_class=HTMLResponse)
def help_page(request: Request, admin: dict = Depends(require_web_admin)):
    return templates.TemplateResponse(request, "help.html", {
        "admin": admin, "nav": "help",
    })


# ----------------------------------------------------------------------- logs --

@router.get("/ui/logs", response_class=HTMLResponse)
def logs_page(request: Request, admin: dict = Depends(require_web_admin)):
    return templates.TemplateResponse(request, "logs.html", {
        "admin": admin, "nav": "logs",
    })


@router.get("/ui/logs/data", response_class=HTMLResponse)
def logs_data(request: Request, admin: dict = Depends(require_web_admin),
              level: str = "ALL", q: str = "", limit: int = 200):
    rows = logbuffer.records(level=level, contains=q or None, limit=limit)
    # Background failures are recorded in the DB rather than logged, so show
    # them alongside — otherwise a failed inference is invisible here.
    try:
        errs = db.query(
            "SELECT kind, detail, created_at FROM errors"
            " ORDER BY created_at DESC LIMIT 20"
        )
    except Exception:
        errs = []
    return templates.TemplateResponse(request, "_logs.html", {
        "rows": rows, "errors": errs, "level": level, "q": q,
        "total": len(logbuffer._records),
    })


@router.post("/ui/logs/clear", response_class=HTMLResponse)
def logs_clear(request: Request, admin: dict = Depends(require_web_admin)):
    logbuffer.clear()
    return templates.TemplateResponse(request, "_logs.html", {
        "rows": [], "errors": [], "level": "ALL", "q": "", "total": 0,
    })


# ----------------------------------------------------------------- performance --

@router.get("/ui/perf", response_class=HTMLResponse)
def perf_page(request: Request, admin: dict = Depends(require_web_admin)):
    # The page shell; the numbers live in a fragment that polls itself.
    return templates.TemplateResponse(request, "perf.html", {
        "admin": admin, "nav": "perf",
    })


@router.get("/ui/perf/data", response_class=HTMLResponse)
def perf_data(request: Request, admin: dict = Depends(require_web_admin)):
    # HTMX does not swap on a 4xx/5xx, so an unhandled error here would leave
    # the page reading "Loading metrics…" forever. Always return 200 with
    # whatever we managed to collect, and say so when a part failed.
    # Defaults carry the sub-keys the template walks into, so a failed
    # collection renders blanks instead of raising UndefinedError.
    ctx: dict = {
        "sys": {"memory": {}, "disk": {}, "temperatures": []},
        "models": [], "tp": None, "errors": [],
    }
    for key, fn in (("sys", metrics.system),
                    ("models", metrics.model_performance),
                    ("tp", metrics.throughput)):
        try:
            ctx[key] = fn()
        except Exception as exc:
            ctx["errors"].append(f"{key}: {type(exc).__name__}: {exc}")
    # system() collects per-reading, so surface those individually too.
    ctx["errors"].extend(ctx["sys"].get("errors", []))
    return templates.TemplateResponse(request, "_perf.html", ctx)


# ----------------------------------------------------------------- dashboard --

@router.get("/", response_class=HTMLResponse)
def root(request: Request):
    return RedirectResponse("/ui", status_code=303)


@router.get("/ui", response_class=HTMLResponse)
def dashboard(request: Request, admin: dict = Depends(require_web_admin)):
    counts = db.query(
        "SELECT (SELECT count(*) FROM boards) AS boards,"
        " (SELECT count(*) FROM views) AS views,"
        " (SELECT count(*) FROM predictions WHERE source='server') AS server_preds,"
        " (SELECT count(*) FROM devices) AS devices,"
        # Not "keys": Jinja's foo.keys would resolve to dict.keys(), not the
        # value. Any dict-method name (keys/values/items/get) is a footgun.
        " (SELECT count(*) FROM api_keys WHERE revoked_at IS NULL) AS active_keys"
    )[0]
    active = db.query(
        "SELECT model_id, version, task, classes FROM models WHERE active"
        " ORDER BY task"
    )
    errors = db.query(
        "SELECT kind, detail, created_at FROM errors ORDER BY created_at DESC LIMIT 8"
    )
    loaded = next(
        (m for m in db.query("SELECT model_id, version FROM models m"
                             " WHERE %s = m.id::text",
                             (runner.model_uuid or "",))), None
    ) if runner.model_uuid else None
    return templates.TemplateResponse(request, "dashboard.html", {
        "admin": admin, "counts": counts, "active": active,
        "errors": errors, "loaded": loaded, "device": runner.device,
        "nav": "dashboard",
    })


# -------------------------------------------------------------------- models --

def _models_rows() -> list[dict]:
    rows = db.query(
        "SELECT id, model_id, version, task, classes, active, registered_at,"
        " meta->'val_metrics' AS val_metrics, meta->>'variant' AS variant"
        " FROM models ORDER BY task, registered_at DESC"
    )
    for r in rows:
        r["id"] = str(r["id"])
        r["loaded"] = runner.is_loaded(r["id"])
    return rows


@router.get("/ui/models", response_class=HTMLResponse)
def models_page(request: Request, admin: dict = Depends(require_web_admin)):
    return templates.TemplateResponse(request, "models.html", {
        "admin": admin, "models": _models_rows(), "nav": "models",
        "notice": None, "notice_kind": "ok",
    })


def _models_fragment(request: Request, notice=None, kind="ok"):
    return templates.TemplateResponse(request, "_models_table.html", {
        "models": _models_rows(), "notice": notice, "notice_kind": kind,
    })


@router.post("/ui/models/register", response_class=HTMLResponse)
async def ui_register(
    request: Request, admin: dict = Depends(require_web_admin),
    archive: UploadFile = File(...), model_id: str = Form(...),
    version: str = Form(...), activate: bool = Form(False),
    metadata: str = Form(""),
):
    try:
        info = routes.save_and_register(
            await archive.read(), model_id.strip(), version.strip(), activate,
            filename=archive.filename, metadata=metadata or None,
        )
        note = f"Registered {info['model_id']}:{info['version']} " \
               f"({info['task']}, {len(info['classes'])} classes)."
        return _models_fragment(request, note, "ok")
    except Exception as exc:
        # HTTPException details already read "rejected: ..." — don't stutter.
        detail = str(getattr(exc, "detail", exc)).removeprefix("rejected: ")
        return _models_fragment(request, f"Rejected: {detail}", "err")


@router.post("/ui/models/try", response_class=HTMLResponse)
async def ui_try_model(
    request: Request, admin: dict = Depends(require_web_admin),
    images: list[UploadFile] = File(...), model_uuid: str = Form(...),
    tta: bool = Form(False),
):
    """Run one model on uploaded image(s) and show the result. Stores nothing —
    this is for checking a model works, not for capturing field data."""
    rows = db.query(
        "SELECT id, model_id, version, task, archive_key, classes, meta"
        " FROM models WHERE id = %s", (model_uuid,)
    )
    if not rows:
        return templates.TemplateResponse(
            request, "_try_result.html", {"error": "No such model."})
    model = rows[0]
    try:
        if not runner.is_loaded(str(model["id"])):
            routes.load_model_row(model, str(model["id"]))
        payload = [await f.read() for f in images]
        out = runner.predict(payload, tta=tta)
        return templates.TemplateResponse(request, "_try_result.html", {
            "out": out, "model": model, "n_images": len(payload),
            "backend": runner.backend, "provider": runner.provider, "error": None,
        })
    except Exception as exc:
        return templates.TemplateResponse(
            request, "_try_result.html", {"error": str(exc)})


@router.post("/ui/models/{model_uuid}/{action}", response_class=HTMLResponse)
def ui_model_action(
    request: Request, model_uuid: str, action: str,
    admin: dict = Depends(require_web_admin),
):
    try:
        if action == "activate":
            routes._activate(model_uuid)
            note = "Activated."
        elif action == "load":
            row = db.query("SELECT archive_key, meta FROM models WHERE id=%s",
                           (model_uuid,))
            if not row:
                return _models_fragment(request, "No such model.", "err")
            routes.load_model_row(row[0], model_uuid)
            note = f"Loaded into memory ({runner.backend}" \
                   f"{', ' + runner.provider if runner.provider else ''})."
        elif action == "unload":
            if runner.is_loaded(model_uuid):
                runner.unload()
            note = "Unloaded."
        else:
            return _models_fragment(request, f"Unknown action {action}.", "err")
        return _models_fragment(request, note, "ok")
    except Exception as exc:
        return _models_fragment(request, f"Failed: {exc}", "err")


# ---------------------------------------------------------------------- keys --

def _keys_rows() -> list[dict]:
    return db.query(
        "SELECT k.id, k.name, k.scope, k.device_id, k.created_at, k.last_used_at,"
        " k.revoked_at FROM api_keys k ORDER BY k.revoked_at NULLS FIRST, k.created_at DESC"
    )


@router.get("/ui/keys", response_class=HTMLResponse)
def keys_page(request: Request, admin: dict = Depends(require_web_admin)):
    return templates.TemplateResponse(request, "keys.html", {
        "admin": admin, "keys": _keys_rows(), "nav": "keys",
        "new_key": None, "notice": None, "notice_kind": "ok",
    })


def _keys_fragment(request: Request, new_key=None, notice=None, kind="ok"):
    return templates.TemplateResponse(request, "_keys_table.html", {
        "keys": _keys_rows(), "new_key": new_key,
        "notice": notice, "notice_kind": kind,
    })


@router.post("/ui/keys/create", response_class=HTMLResponse)
def ui_create_key(
    request: Request, admin: dict = Depends(require_web_admin),
    name: str = Form(...), scope: str = Form(...), device_id: str = Form(""),
):
    scope = scope.strip()
    device_id = device_id.strip() or None
    if scope not in ("admin", "ingest"):
        return _keys_fragment(request, None, "Scope must be admin or ingest.", "err")
    if scope == "ingest" and not device_id:
        return _keys_fragment(request, None, "Ingest keys need a device id.", "err")

    if device_id:
        db.execute(
            "INSERT INTO devices (device_id) VALUES (%s) ON CONFLICT DO NOTHING",
            (device_id,),
        )
    token, digest = auth.new_key()
    db.execute(
        "INSERT INTO api_keys (name, key_hash, scope, device_id) VALUES (%s,%s,%s,%s)",
        (name.strip(), digest, scope, device_id),
    )
    # The plaintext is shown once, right here, and never stored.
    return _keys_fragment(request, token, f"Created '{name.strip()}'. Copy it now.", "ok")


@router.post("/ui/keys/{key_id}/revoke", response_class=HTMLResponse)
def ui_revoke_key(
    request: Request, key_id: int, admin: dict = Depends(require_web_admin),
):
    # A key cannot revoke itself out from under the session that's using it.
    if str(key_id) == str(admin["id"]):
        return _keys_fragment(request, None, "Can't revoke the key you're logged in with.", "err")
    db.execute("UPDATE api_keys SET revoked_at = now() WHERE id = %s", (key_id,))
    return _keys_fragment(request, None, "Revoked.", "ok")
