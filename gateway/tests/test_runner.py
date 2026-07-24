"""Model-agnosticism and contract tests.

The point of this file: two archives with different tasks, class counts, input
sizes, view counts, and output shapes must both flow through the same runner
with no branch anywhere that names either of them.
"""

import io
import json
import random
from pathlib import Path

import pytest
from PIL import Image

from app.runner import ManifestError, ModelRunner, parse_metadata

MODELS = Path("/models")
BOARD = MODELS / "seed_board_clf.ts.pt"
DETECTOR = MODELS / "seed_detector.ts.pt"

pytestmark = pytest.mark.skipif(
    not BOARD.exists(), reason="seed models not built; run scripts/make_seed_models.py"
)


def png_bytes(w: int = 900, h: int = 700) -> bytes:
    """A lossless test capture. Deterministic, and large enough that patch
    mode has room for a real 2x2 grid at 384px.

    Built without numpy on purpose: the L4T PyTorch image ships a numpy whose
    ABI is pinned to its torch build, so this package must not depend on one.
    """
    rng = random.Random(1)
    raw = bytes(rng.randrange(256) for _ in range(w * h * 3))
    buf = io.BytesIO()
    Image.frombytes("RGB", (w, h), raw).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def runner():
    r = ModelRunner()
    yield r
    r.unload()


# ------------------------------------------------------------------ manifest --

def test_manifest_defaults_for_older_archives():
    """An archive predating task types is a classifier, single-view."""
    meta = parse_metadata(json.dumps({
        "classes": ["a", "b"],
        "image_size": 224,
        "normalize_mean": [0.5, 0.5, 0.5],
        "normalize_std": [0.5, 0.5, 0.5],
    }))
    assert meta["task"] == "classification"
    assert meta["multi_view"] is False
    assert meta["input_layout"] == "image"
    assert meta["temperature"] == 1.0


@pytest.mark.parametrize("bad,reason", [
    ({}, "missing everything"),
    ({"classes": [], "image_size": 1, "normalize_mean": [0] * 3,
      "normalize_std": [1] * 3}, "empty class list"),
    ({"classes": ["a"], "image_size": 1, "normalize_mean": [0] * 2,
      "normalize_std": [1] * 3}, "wrong mean length"),
    ({"classes": ["a"], "image_size": 1, "task": "telepathy",
      "normalize_mean": [0] * 3, "normalize_std": [1] * 3}, "unknown task"),
])
def test_bad_manifests_are_rejected(bad, reason):
    with pytest.raises(ManifestError):
        parse_metadata(json.dumps(bad))


def test_missing_metadata_is_rejected():
    with pytest.raises(ManifestError):
        parse_metadata("")


# ----------------------------------------------------------- the board model --

def test_board_model_full_three_views(runner):
    meta = runner.load(BOARD, "board-1")
    assert meta["max_views"] == 3 and meta["patch_mode"] is True

    out = runner.predict([png_bytes(), png_bytes(), png_bytes()])

    assert set(out["probs"]) == set(meta["classes"])
    assert out["label"] in meta["classes"]
    assert abs(sum(out["probs"].values()) - 1.0) < 1e-5
    assert len(out["view_attention"]) == 3
    # Attention is a distribution over the submitted views.
    assert abs(sum(out["view_attention"]) - 1.0) < 1e-3
    # Patch mode declares 4 patches per view.
    assert [len(p) for p in out["patch_attention"]] == [4, 4, 4]
    assert out["calibrated"] is True          # temperature 1.35 in the manifest
    assert out["latency_ms"] > 0


@pytest.mark.parametrize("n_views", [1, 2, 3])
def test_partial_boards(runner, n_views):
    """One, two, or three views all work. The pooler is view-count agnostic,
    so partial uploads must not be special-cased anywhere."""
    runner.load(BOARD, "board-1")
    out = runner.predict([png_bytes() for _ in range(n_views)])

    assert len(out["view_attention"]) == n_views
    assert abs(sum(out["view_attention"]) - 1.0) < 1e-3
    assert abs(sum(out["probs"].values()) - 1.0) < 1e-5


def test_too_many_views_is_rejected(runner):
    runner.load(BOARD, "board-1")
    with pytest.raises(ValueError):
        runner.predict([png_bytes() for _ in range(4)])


def test_tta_still_returns_a_distribution(runner):
    runner.load(BOARD, "board-1")
    out = runner.predict([png_bytes(), png_bytes()], tta=True)
    assert out["tta"] is True
    assert abs(sum(out["probs"].values()) - 1.0) < 1e-4
    assert set(out["probs"]) == set(runner.meta["classes"])


def test_temperature_is_applied(runner):
    """Calibration is load-bearing: the review queue ranks on confidence, so a
    temperature that silently does nothing would corrupt the ordering."""
    runner.load(BOARD, "board-1")
    img = [png_bytes()]
    calibrated = runner.predict(img)["probs"]

    runner.meta["temperature"] = 1.0
    uncalibrated = runner.predict(img)["probs"]

    assert calibrated != uncalibrated
    # T > 1 flattens the distribution.
    assert max(calibrated.values()) < max(uncalibrated.values())


# -------------------------------------------------------- the OTHER model ----
# Different task, class count, input size, view count, output shape. If any of
# these needs a code change, the abstraction is wrong.

def test_detector_loads_and_predicts(runner):
    meta = runner.load(DETECTOR, "det-1")
    assert meta["task"] == "detection"
    assert meta["multi_view"] is False
    assert meta["input_layout"] == "image"
    assert len(meta["classes"]) == 5

    out = runner.predict([png_bytes()])

    assert out["detections"], "detector returned no boxes"
    for det in out["detections"]:
        assert det["label"] in meta["classes"]
        assert 0.0 <= det["score"] <= 1.0
        x1, y1, x2, y2 = det["box"]
        assert x2 >= x1 and y2 >= y1
    # Sorted most-confident first, so the UI can slice the top N.
    scores = [d["score"] for d in out["detections"]]
    assert scores == sorted(scores, reverse=True)
    assert out["view_attention"] is None      # no pooler, no attention


def test_swapping_models_evicts_the_previous_one(runner):
    """One model resident at a time — 8GB is the hard limit."""
    runner.load(BOARD, "board-1")
    assert runner.is_loaded("board-1")

    runner.load(DETECTOR, "det-1")
    assert runner.is_loaded("det-1")
    assert not runner.is_loaded("board-1")

    runner.unload()
    assert not runner.is_loaded()


def test_predict_without_a_model_fails_loudly(runner):
    with pytest.raises(RuntimeError):
        runner.predict([png_bytes()])


# ------------------------------------------------------------- agnosticism ---

def test_no_class_names_leak_into_server_code():
    """The real regression guard. Grade names live in archive metadata and
    nowhere else — not in code, not in templates, not in the schema.

    The seed manifests deliberately use fake class names, so a leak from
    either archive shows up here.
    """
    leaked = ["2A", "3A", "4A", "gradeA", "gradeB", "gradeC"]
    roots = [Path("/app/app"), Path("/app/migrations")]
    offenders = []
    for root in roots:
        for path in root.rglob("*"):
            if path.suffix not in (".py", ".sql", ".html") or not path.is_file():
                continue
            text = path.read_text(encoding="utf-8")
            for name in leaked:
                # Word-ish check: "2A" must not appear as a bare token.
                if name in text:
                    offenders.append(f"{path}: {name}")
    assert not offenders, f"class names hardcoded in server code: {offenders}"
