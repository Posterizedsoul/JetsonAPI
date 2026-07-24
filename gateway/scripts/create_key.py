"""Mint an API key. Run inside the gateway container:

    docker compose exec gateway python /app/scripts/create_key.py admin "laptop"
    docker compose exec gateway python /app/scripts/create_key.py ingest "phone-01" --device phone-01

The plaintext is printed once and is not recoverable afterwards.
"""

import argparse
import sys

sys.path.insert(0, "/app")

from app import auth, db  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("scope", choices=["admin", "ingest"])
    ap.add_argument("name")
    ap.add_argument("--device", help="device_id, required for ingest keys")
    args = ap.parse_args()

    if args.scope == "ingest" and not args.device:
        ap.error("ingest keys must be bound to a device: --device <id>")

    db.pool.open(wait=True, timeout=30)
    if args.device:
        db.execute(
            "INSERT INTO devices (device_id) VALUES (%s) ON CONFLICT DO NOTHING",
            (args.device,),
        )
    token, digest = auth.new_key()
    db.execute(
        "INSERT INTO api_keys (name, key_hash, scope, device_id) VALUES (%s,%s,%s,%s)",
        (args.name, digest, args.scope, args.device),
    )
    print(f"\nscope:  {args.scope}\ndevice: {args.device or '-'}\nkey:    {token}\n")
    print("Store it now — only the hash is kept.")


if __name__ == "__main__":
    main()
