"""TorchScript model loading and inference.

WHERE MODEL-SPECIFIC KNOWLEDGE LIVES
------------------------------------
All of it comes from metadata.json embedded in the archive via _extra_files.
Nothing in this file, the database, or the templates knows what a class is
called, how big the input is, or whether patches are used. A five-class model
needs no code change and no migration.

The one genuine boundary: a new *task type* needs a postprocess function here
and a renderer in the UI. A new *model of an existing task* needs neither —
that is what makes the second seed model a real test rather than a formality.
"""

import io
import json
import math
import threading
import time
from pathlib import Path

import torch
from PIL import Image

from app import config

# Defaults for keys older archives predate. Absent 'task' means the archive was
# exported before task types existed, and every such archive is a classifier.
METADATA_DEFAULTS = {
    "task": "classification",
    "max_views": 1,
    "patch_mode": False,
    "patches_per_view": 4,
    "temperature": 1.0,
    "variant": None,
    "val_metrics": {},
}
REQUIRED_KEYS = ("classes", "image_size", "normalize_mean", "normalize_std")

_TTA_FLIPS = ((False, False), (True, False), (False, True), (True, True))


class ManifestError(ValueError):
    """Archive metadata is missing or malformed. Registration rejects it."""


def backend_of(archive_path: str | Path) -> str:
    """Which runtime serves this file. Extension is the whole rule — no
    sniffing, no registry: '.onnx' is ONNX Runtime, anything else is
    TorchScript."""
    return "onnx" if str(archive_path).lower().endswith(".onnx") else "torchscript"


def _onnx_runtime():
    """Import ONNX Runtime, turning a missing install into a clear message
    instead of a raw ModuleNotFoundError surfacing as a 500."""
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ManifestError(
            "ONNX Runtime is not installed in the gateway container, so .onnx "
            "models cannot be served. Rebuild the image: "
            "sudo ./setup-jetson.sh deploy"
        ) from exc
    return ort


def _onnx_embedded_metadata(archive_path: str | Path) -> str:
    """ONNX's equivalent of TorchScript's _extra_files is metadata_props, a
    string->string map on the model. We look for a 'metadata.json' key so the
    same manifest travels inside either archive format."""
    ort = _onnx_runtime()

    try:
        opts = ort.SessionOptions()
        # Reading metadata only: skip graph optimization so this stays cheap
        # and can't fail on a model the CPU provider would rewrite.
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        sess = ort.InferenceSession(str(archive_path), opts,
                                    providers=["CPUExecutionProvider"])
    except Exception as exc:
        raise ManifestError(f"not a loadable ONNX model: {exc}") from exc
    props = sess.get_modelmeta().custom_metadata_map or {}
    return props.get("metadata.json") or props.get("metadata") or ""


def read_metadata(archive_path: str | Path, fallback: str | None = None) -> dict:
    """Load and validate the manifest without keeping the model resident.

    `fallback` is metadata supplied at registration time. It is only consulted
    when the archive carries none of its own — the embedded copy always wins,
    so a self-describing archive can never be overridden by a stale paste.
    Stock ONNX exports have no embedded metadata, which is why the fallback
    exists at all; TorchScript archives from export.py always embed it.
    """
    if backend_of(archive_path) == "onnx":
        raw = _onnx_embedded_metadata(archive_path)
    else:
        extra = {"metadata.json": ""}
        try:
            torch.jit.load(str(archive_path), map_location="cpu", _extra_files=extra)
        except Exception as exc:
            raise ManifestError(f"not a loadable TorchScript archive: {exc}") from exc
        raw = extra.get("metadata.json") or ""

    if not raw:
        raw = (fallback or "").strip()
        if not raw:
            raise ManifestError(
                "no metadata found in the archive. TorchScript archives should "
                "embed metadata.json via _extra_files; ONNX models via "
                "metadata_props. For an ONNX export without it, paste the "
                "manifest JSON in the Metadata field when registering."
            )
    return parse_metadata(raw)


def parse_metadata(raw: str | bytes) -> dict:
    if not raw:
        raise ManifestError("archive has no embedded metadata.json")
    try:
        meta = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ManifestError(f"metadata.json is not valid JSON: {exc}") from exc

    missing = [k for k in REQUIRED_KEYS if k not in meta]
    if missing:
        raise ManifestError(f"metadata.json missing required keys: {missing}")
    if not isinstance(meta["classes"], list) or not meta["classes"]:
        raise ManifestError("metadata.json 'classes' must be a non-empty list")
    if not all(isinstance(c, str) for c in meta["classes"]):
        raise ManifestError("metadata.json 'classes' must be strings")
    for key in ("normalize_mean", "normalize_std"):
        if len(meta[key]) != 3:
            raise ManifestError(f"metadata.json '{key}' must have 3 values")

    for key, default in METADATA_DEFAULTS.items():
        meta.setdefault(key, default)

    # Derived, so archives need not spell them out. GibsonNet declares
    # max_views=3 and is therefore a board model; a plain vision model
    # declares 1 and takes a single image tensor.
    meta.setdefault("multi_view", meta["max_views"] > 1)
    meta.setdefault("input_layout", "board" if meta["multi_view"] else "image")

    if meta["task"] not in POSTPROCESS:
        raise ManifestError(
            f"unknown task {meta['task']!r}; known: {sorted(POSTPROCESS)}"
        )
    if meta["input_layout"] not in ("board", "image"):
        raise ManifestError(f"unknown input_layout {meta['input_layout']!r}")
    return meta


def grid_boxes(width: int, height: int, size: int, k: int) -> list[tuple]:
    """Deterministic K-crop grid. Mirrors gibsonnet.data.grid_boxes exactly —
    serving preprocessing must match evaluation or the metrics lie."""
    cols = max(1, round(k ** 0.5))
    rows = (k + cols - 1) // cols
    boxes = []
    for r in range(rows):
        for c in range(cols):
            if len(boxes) == k:
                break
            x = round(c * (width - size) / max(cols - 1, 1)) if cols > 1 else (width - size) // 2
            y = round(r * (height - size) / max(rows - 1, 1)) if rows > 1 else (height - size) // 2
            boxes.append((x, y, x + size, y + size))
    return boxes


# --------------------------------------------------------------- postprocess --
# One function per task type. The dispatch table is the extension point: a new
# task adds an entry here and a template partial, nothing else.

def _classification(raw, meta: dict) -> dict:
    logits, attention = raw
    temperature = meta["temperature"]
    probs = torch.softmax(logits[0].float() / temperature, dim=0).tolist()
    classes = meta["classes"]
    order = sorted(range(len(probs)), key=probs.__getitem__, reverse=True)
    top, second = order[0], (order[1] if len(order) > 1 else order[0])
    return {
        "probs": dict(zip(classes, probs)),
        "label": classes[top],
        "confidence": probs[top],
        # Narrow margin = the model is torn between two grades; the review
        # queue sorts on this, so it is stored rather than recomputed.
        "margin": probs[top] - probs[second] if len(probs) > 1 else 1.0,
        "attention": attention[0].float().tolist() if attention is not None else None,
        "calibrated": temperature != 1.0,
    }


def _detection(raw, meta: dict) -> dict:
    # torchvision detection convention: dict of boxes/scores/labels, or a
    # tuple in that order. Boxes are xyxy in input-tensor pixels.
    out = raw[0] if isinstance(raw, (list, tuple)) and len(raw) == 1 else raw
    if isinstance(out, dict):
        boxes, scores, labels = out["boxes"], out["scores"], out["labels"]
    else:
        boxes, scores, labels = out[:3]
    classes = meta["classes"]
    dets = [
        {
            "box": [round(v, 2) for v in box.tolist()],
            "score": float(score),
            "label": classes[int(lab)] if int(lab) < len(classes) else str(int(lab)),
        }
        for box, score, lab in zip(boxes, scores, labels)
    ]
    dets.sort(key=lambda d: d["score"], reverse=True)
    return {
        "detections": dets,
        "label": dets[0]["label"] if dets else None,
        "confidence": dets[0]["score"] if dets else 0.0,
        "margin": None,
        "probs": {},
        "attention": None,
    }


def _segmentation(raw, meta: dict) -> dict:
    logits = raw["out"] if isinstance(raw, dict) else (raw[0] if isinstance(raw, (list, tuple)) else raw)
    mask = logits[0].argmax(0).to(torch.uint8)
    classes = meta["classes"]
    present = mask.unique().tolist()
    counts = {classes[i]: int((mask == i).sum()) for i in present if i < len(classes)}
    total = int(mask.numel())
    return {
        # The mask itself is large; the UI overlays it from a separate
        # endpoint. Here we keep the summary the review queue sorts on.
        "coverage": {k: v / total for k, v in counts.items()},
        "label": max(counts, key=counts.get) if counts else None,
        "confidence": max(counts.values()) / total if counts else 0.0,
        "margin": None,
        "probs": {},
        "attention": None,
        "mask_shape": list(mask.shape),
    }


POSTPROCESS = {
    "classification": _classification,
    "detection": _detection,
    "segmentation": _segmentation,
}


# ------------------------------------------------------------------- runner --

class ModelRunner:
    """One resident model at a time — 8GB unified memory is the hard limit.

    Loading a different archive unloads the current one first. Cold-load
    latency is acceptable because nothing synchronous waits on inference.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.model = None            # TorchScript module, or ORT session
        self.backend = "torchscript"
        self.provider: str | None = None   # which ORT execution provider won
        self._onnx_inputs: list = []
        self.meta: dict = {}
        self.model_uuid: str | None = None
        self.device = config.DEVICE or ("cuda" if torch.cuda.is_available() else "cpu")
        # fp16 on the Jetson roughly halves latency and memory. CPU stays fp32
        # because half-precision CPU kernels are slower, not faster.
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

    def is_loaded(self, model_uuid: str | None = None) -> bool:
        if self.model is None:
            return False
        return model_uuid is None or self.model_uuid == model_uuid

    def load(self, archive_path: str | Path, model_uuid: str,
             fallback_meta: str | None = None) -> dict:
        with self._lock:
            if self.model_uuid == model_uuid and self.model is not None:
                return self.meta
            self._unload_locked()
            if backend_of(archive_path) == "onnx":
                meta = self._load_onnx(archive_path, fallback_meta)
            else:
                meta = self._load_torchscript(archive_path)
            self.meta = meta
            self.model_uuid = model_uuid
            return meta

    def _load_torchscript(self, archive_path) -> dict:
        extra = {"metadata.json": ""}
        model = torch.jit.load(
            str(archive_path), map_location=self.device, _extra_files=extra
        )
        model.eval()
        meta = parse_metadata(extra.get("metadata.json") or "")
        if self.device == "cuda":
            model = model.to(memory_format=torch.channels_last).half()
        self.model = model
        self.backend = "torchscript"
        self.provider = None
        return meta

    def _load_onnx(self, archive_path, fallback_meta: str | None) -> dict:
        ort = _onnx_runtime()

        raw = _onnx_embedded_metadata(archive_path) or (fallback_meta or "")
        meta = parse_metadata(raw)

        # Preference order, first available wins. TensorRT is skipped on
        # purpose: it recompiles the graph on first run, which can take
        # minutes and would look like a hang on load.
        available = ort.get_available_providers()
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                     if p in available]
        sess = ort.InferenceSession(str(archive_path), providers=providers)
        self.model = sess
        self.backend = "onnx"
        self.provider = (sess.get_providers() or ["unknown"])[0]
        self._onnx_inputs = sess.get_inputs()
        return meta

    def unload(self) -> None:
        with self._lock:
            self._unload_locked()

    def _unload_locked(self) -> None:
        self.model = None
        self.provider = None
        self._onnx_inputs = []
        self.meta = {}
        self.model_uuid = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -------------------------------------------------------- preprocessing --

    def _to_tensor(self, img: Image.Image) -> torch.Tensor:
        mean = torch.tensor(self.meta["normalize_mean"]).view(3, 1, 1)
        std = torch.tensor(self.meta["normalize_std"]).view(3, 1, 1)
        t = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8)
        t = t.view(img.height, img.width, 3).permute(2, 0, 1).float() / 255.0
        return (t - mean) / std

    def preprocess_view(self, data: bytes) -> list[torch.Tensor]:
        """One view's bytes -> its instances (1 whole-face, or K patches).

        Reads the ORIGINAL lossless bytes. Previews are never used here.
        """
        meta = self.meta
        size = meta["image_size"]
        with Image.open(io.BytesIO(data)) as raw:
            img = raw.convert("RGB")
            if meta["patch_mode"]:
                if min(img.size) < size:
                    scale = size / min(img.size)
                    img = img.resize(
                        (max(size, round(img.width * scale)),
                         max(size, round(img.height * scale))),
                        Image.BICUBIC,
                    )
                return [
                    self._to_tensor(img.crop(box))
                    for box in grid_boxes(img.width, img.height, size,
                                          meta["patches_per_view"])
                ]
            # Whole-face: short side to 1.15x target, centre crop — the eval
            # transform from gibsonnet.data.build_eval_transform.
            scale = int(size * 1.15) / min(img.size)
            img = img.resize(
                (round(img.width * scale), round(img.height * scale)), Image.BICUBIC
            )
            left = (img.width - size) // 2
            top = (img.height - size) // 2
            return [self._to_tensor(img.crop((left, top, left + size, top + size)))]

    # ------------------------------------------------------------ inference --

    def predict(self, views: list[bytes], tta: bool = False) -> dict:
        """Run one board (or one image, for single-view models)."""
        if self.model is None:
            raise RuntimeError("no model loaded")
        meta = self.meta
        max_views = meta["max_views"]
        if not 1 <= len(views) <= max_views:
            raise ValueError(f"expected 1..{max_views} views, got {len(views)}")

        started = time.perf_counter()
        per_view = [self.preprocess_view(v) for v in views]
        k = len(per_view[0])
        instances = [inst for view in per_view for inst in view]

        with self._lock:
            batch = torch.stack(instances).unsqueeze(0)  # (1, V*K, 3, S, S)
            if meta["input_layout"] == "image":
                batch = batch[0]                          # (V*K, 3, S, S)
            raw = self._forward(batch, len(instances), tta)

        result = POSTPROCESS[meta["task"]](raw, meta)
        result["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
        result["tta"] = tta

        # Collapse per-instance attention back to per-photo numbers: in patch
        # mode the pooler attends over V*K instances, but the operator asked
        # "which lighting condition drove this", not "which patch".
        attn = result.pop("attention", None)
        if attn is not None:
            result["view_attention"] = [
                round(sum(attn[i * k:(i + 1) * k]), 4) for i in range(len(per_view))
            ]
            if meta["patch_mode"]:
                result["patch_attention"] = [
                    [round(a, 4) for a in attn[i * k:(i + 1) * k]]
                    for i in range(len(per_view))
                ]
        else:
            result["view_attention"] = None
        return result

    def _onnx_run(self, batch: torch.Tensor, n_instances: int, tta: bool):
        """ONNX Runtime path. Feeds the graph's declared inputs positionally:
        first input is the image batch, an optional second is the view mask
        (multi-view models). Outputs come back as numpy and are wrapped in
        torch tensors so the postprocess functions stay backend-agnostic."""
        meta = self.meta
        # ORT wants exactly the dtype the graph declares — no autocasting.
        want16 = bool(self._onnx_inputs) and "float16" in self._onnx_inputs[0].type
        arr = batch.to(torch.float16 if want16 else torch.float32).cpu().numpy()

        feed = {self._onnx_inputs[0].name: arr}
        if len(self._onnx_inputs) > 1:
            import numpy as np
            second = self._onnx_inputs[1]
            dtype = np.bool_ if "bool" in second.type else np.float32
            feed[second.name] = np.ones((1, n_instances), dtype=dtype)

        outs = [torch.from_numpy(o) if hasattr(o, "dtype") else o
                for o in self.model.run(None, feed)]

        if tta:
            # Same flip ensemble as the torch path, run through the session.
            probs = torch.softmax(outs[0].float() / meta["temperature"], dim=1)
            for h, v in _TTA_FLIPS[1:]:
                dims = [d for d, f in zip((-1, -2), (h, v)) if f]
                flipped = dict(feed)
                flipped[self._onnx_inputs[0].name] = (
                    batch.flip(dims=dims)
                    .to(torch.float16 if want16 else torch.float32).cpu().numpy()
                )
                out = torch.from_numpy(self.model.run(None, flipped)[0])
                probs = probs + torch.softmax(
                    out.float() / meta["temperature"], dim=1)
            probs = probs / len(_TTA_FLIPS)
            logits = torch.log(probs.clamp_min(1e-12)) * meta["temperature"]
            return logits, (outs[1] if len(outs) > 1 else None)

        if meta["task"] == "classification":
            # Second output, when present, is the attention weights.
            return outs[0], (outs[1] if len(outs) > 1 else None)
        return tuple(outs) if len(outs) > 1 else outs[0]

    def _forward(self, batch: torch.Tensor, n_instances: int, tta: bool):
        meta = self.meta
        if self.backend == "onnx":
            return self._onnx_run(batch, n_instances, tta)
        with torch.inference_mode():
            batch = batch.to(self.device, dtype=self.dtype)
            # channels_last is defined for rank-4 NCHW only. A multi-view board
            # batch is (1, V*K, 3, H, W) — rank 5 — and asking for the format
            # there raises "required rank 4 tensor to use channels_last".
            # The model folds views into the batch internally, so the encoder
            # still sees 4-D; we just can't pre-format the outer tensor.
            if self.device == "cuda" and batch.dim() == 4:
                batch = batch.contiguous(memory_format=torch.channels_last)

            # Only real instances are sent, so every slot is a real one. The
            # reference server padded to max_views*K and burned ~3x the encoder
            # compute on zeros for single-view boards; masked slots get exactly
            # zero attention weight, so dropping them is numerically identical.
            mask = torch.ones(1, n_instances, dtype=torch.bool, device=self.device)

            if meta["input_layout"] == "board":
                raw = self.model.forward_with_attention(batch, mask)
                if not tta:
                    return raw
                logits, attention = raw
                probs = torch.softmax(logits.float() / meta["temperature"], dim=1)
                for h, v in _TTA_FLIPS[1:]:
                    dims = [d for d, f in zip((-1, -2), (h, v)) if f]
                    flipped = self.model(batch.flip(dims=dims), mask)
                    probs = probs + torch.softmax(
                        flipped.float() / meta["temperature"], dim=1
                    )
                probs = probs / len(_TTA_FLIPS)
                # Hand back logits-shaped values so postprocess stays uniform:
                # log of an averaged distribution, temperature already applied.
                return torch.log(probs.clamp_min(1e-12)) * meta["temperature"], attention

            out = self.model(batch)
            return (out, None) if meta["task"] == "classification" else out


runner = ModelRunner()
