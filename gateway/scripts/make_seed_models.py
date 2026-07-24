"""Build two TorchScript archives that exercise the whole serving path.

    python scripts/make_seed_models.py --out data/models

Neither has trained weights — they exist so the pipeline, the review UI, and
the model-agnosticism test can be verified before a real export lands.

  seed_board_clf   mirrors GibsonNet exactly: multi-view (max_views=3), gated
                   attention with forward_with_attention, patch mode on,
                   3 classes, 384px, calibration temperature.

  seed_detector    deliberately unlike it: detection task, 1 view, 5 classes,
                   256px, no attention, no patches. Registering this must
                   require zero code changes — that is the whole point.

Swap in a real archive by pointing the register call at it; nothing here is
referenced by the server at runtime.
"""

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch import Tensor, nn


class TinyEncoder(nn.Module):
    """Stand-in for a pretrained backbone: (N,3,H,W) -> (N,D)."""

    def __init__(self, dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, dim, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class SeedBoardNet(nn.Module):
    """Same public interface as gibsonnet.model.WoodGradeNet.

    The server calls forward_with_attention(views, mask) and expects
    (logits, weights) with padded slots at exactly zero weight.
    """

    def __init__(self, num_classes: int, dim: int = 64) -> None:
        super().__init__()
        self.encoder = TinyEncoder(dim)
        self.tanh_proj = nn.Linear(dim, 32)
        self.gate_proj = nn.Linear(dim, 32)
        self.score = nn.Linear(32, 1)
        self.head = nn.Sequential(
            nn.LayerNorm(dim), nn.Dropout(0.0), nn.Linear(dim, num_classes)
        )

    def _pool(self, feats: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        scores = self.score(
            torch.tanh(self.tanh_proj(feats)) * torch.sigmoid(self.gate_proj(feats))
        ).squeeze(-1)
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=1)
        fused = torch.bmm(weights.unsqueeze(1), feats).squeeze(1)
        return fused, weights

    def forward(self, views: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        b, v = views.shape[0], views.shape[1]
        if mask is None:
            mask = torch.ones(b, v, dtype=torch.bool, device=views.device)
        feats = self.encoder(views.flatten(0, 1)).reshape(b, v, -1)
        fused, _ = self._pool(feats, mask)
        return self.head(fused)

    @torch.jit.export
    def forward_with_attention(
        self, views: Tensor, mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        b, v = views.shape[0], views.shape[1]
        if mask is None:
            mask = torch.ones(b, v, dtype=torch.bool, device=views.device)
        feats = self.encoder(views.flatten(0, 1)).reshape(b, v, -1)
        fused, weights = self._pool(feats, mask)
        return self.head(fused), weights


class SeedDetector(nn.Module):
    """Single-image detector: (N,3,H,W) -> boxes/scores/labels.

    Structurally unlike SeedBoardNet in every dimension the server might have
    been tempted to hardcode: no views, no mask, no attention, no patches,
    different class count, different input size, different output shape.
    """

    def __init__(self, num_classes: int, num_boxes: int = 8) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_boxes = num_boxes
        self.encoder = TinyEncoder(64)
        self.box_head = nn.Linear(64, num_boxes * 4)
        self.cls_head = nn.Linear(64, num_boxes * num_classes)

    def forward(self, images: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        feats = self.encoder(images)[:1]           # first image of the batch
        boxes = torch.sigmoid(self.box_head(feats)).reshape(self.num_boxes, 4)
        # xyxy in input pixels, ordered so x1<x2, y1<y2.
        size = float(images.shape[-1])
        xy1 = boxes[:, :2] * size * 0.5
        xy2 = xy1 + boxes[:, 2:] * size * 0.5
        boxes = torch.cat([xy1, xy2], dim=1)
        logits = self.cls_head(feats).reshape(self.num_boxes, self.num_classes)
        probs = torch.softmax(logits, dim=1)
        scores, labels = probs.max(dim=1)
        return boxes, scores, labels


def save(model: nn.Module, meta: dict, path: Path, example) -> None:
    model.eval()
    scripted = torch.jit.script(model)
    with torch.no_grad():
        scripted(*example)          # smoke test at build time, not in prod
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(
        scripted, str(path),
        _extra_files={"metadata.json": json.dumps(meta, indent=2)},
    )
    print(f"wrote {path}  task={meta['task']}  classes={meta['classes']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/models")
    args = ap.parse_args()
    out = Path(args.out)
    torch.manual_seed(0)

    # Placeholder grade names. They are NOT the real 2A/3A/4A set — using
    # obviously fake names here is deliberate: if any of these strings ever
    # show up in server code or a template, the leak is easy to spot.
    board_classes = ["gradeA", "gradeB", "gradeC"]
    save(
        SeedBoardNet(len(board_classes)),
        {
            "classes": board_classes,
            "task": "classification",
            "variant": "seed",
            "image_size": 384,
            "max_views": 3,
            "patch_mode": True,
            "patches_per_view": 4,
            "temperature": 1.35,
            "normalize_mean": [0.485, 0.456, 0.406],
            "normalize_std": [0.229, 0.224, 0.225],
            "val_metrics": {"macro_f1": 0.0, "grade_mae": 0.0, "note": "untrained seed"},
            "source_checkpoint": "seed (untrained)",
        },
        out / "seed_board_clf.ts.pt",
        (torch.randn(1, 6, 3, 384, 384), torch.ones(1, 6, dtype=torch.bool)),
    )

    det_classes = ["knot", "crack", "split", "wane", "resin"]
    save(
        SeedDetector(len(det_classes)),
        {
            "classes": det_classes,
            "task": "detection",
            "variant": "seed-det",
            "image_size": 256,
            "max_views": 1,
            "patch_mode": False,
            "temperature": 1.0,
            "normalize_mean": [0.5, 0.5, 0.5],
            "normalize_std": [0.5, 0.5, 0.5],
            "val_metrics": {"mAP": 0.0, "note": "untrained seed"},
            "source_checkpoint": "seed (untrained)",
        },
        out / "seed_detector.ts.pt",
        (torch.randn(1, 3, 256, 256),),
    )


if __name__ == "__main__":
    main()
