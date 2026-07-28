"""Embed a manifest into an ONNX file so it is self-describing.

Run this on the machine that trained/exported the model (needs `onnx`,
`pip install onnx`) — not on the Jetson.

    python stamp_onnx_metadata.py model.onnx --meta manifest.json
    python stamp_onnx_metadata.py model.onnx --meta manifest.json -o stamped.onnx
    python stamp_onnx_metadata.py model.onnx --show

WHY: TorchScript archives carry metadata.json via _extra_files; ONNX's
equivalent is metadata_props, a string->string map. torch.onnx.export writes
none, so a stock export tells the server nothing about its classes or
preprocessing. Stamping it once means the file can be registered anywhere
without pasting a manifest, and the server can never desynchronize from it.
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import onnx
except ImportError:
    sys.exit("needs the onnx package:  pip install onnx")

# Mirrors gateway/app/runner.py — registration rejects a manifest missing these.
REQUIRED = ("classes", "image_size", "normalize_mean", "normalize_std")


def show(path: Path) -> None:
    model = onnx.load(str(path), load_external_data=False)
    props = {p.key: p.value for p in model.metadata_props}
    if not props:
        print(f"{path.name}: no metadata_props (stock export)")
        return
    for key, value in props.items():
        print(f"--- {key} ---")
        print(value)


def stamp(path: Path, meta_path: Path, out: Path | None) -> None:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    missing = [k for k in REQUIRED if k not in meta]
    if missing:
        sys.exit(f"manifest is missing required keys: {missing}")
    if not isinstance(meta.get("classes"), list) or not meta["classes"]:
        sys.exit("'classes' must be a non-empty list")
    for key in ("normalize_mean", "normalize_std"):
        if len(meta[key]) != 3:
            sys.exit(f"'{key}' must have 3 values")

    model = onnx.load(str(path))
    # Replace any existing entry rather than appending a duplicate key.
    keep = [p for p in model.metadata_props if p.key not in ("metadata.json", "metadata")]
    del model.metadata_props[:]
    model.metadata_props.extend(keep)
    entry = model.metadata_props.add()
    entry.key = "metadata.json"
    entry.value = json.dumps(meta)

    dest = out or path
    onnx.save(model, str(dest))
    print(f"stamped {dest}")
    print(f"  task    : {meta.get('task', 'classification')}")
    print(f"  classes : {meta['classes']}")
    print(f"  input   : {meta['image_size']}px, max_views={meta.get('max_views', 1)}")
    print("\nRegister it with no Metadata field — the file describes itself now.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", type=Path)
    ap.add_argument("--meta", type=Path, help="manifest JSON to embed")
    ap.add_argument("-o", "--out", type=Path, help="write here instead of in place")
    ap.add_argument("--show", action="store_true", help="print existing metadata and exit")
    args = ap.parse_args()

    if not args.model.exists():
        sys.exit(f"no such file: {args.model}")
    if args.show:
        show(args.model)
    elif args.meta:
        stamp(args.model, args.meta, args.out)
    else:
        ap.error("pass --meta manifest.json, or --show")


if __name__ == "__main__":
    main()
