"""API key auth. Keys are stored as sha256 hashes; the plaintext is shown
once at creation and never again.

Ingest keys are bound to a device, so device_id is derived from the key rather
than trusted from the request body — a buggy client cannot attribute its
uploads to another device.
"""

import hashlib
import secrets

from fastapi import Depends, Header, HTTPException

from app import db


def hash_key(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def new_key() -> tuple[str, str]:
    """(plaintext, hash). Show the plaintext once."""
    token = secrets.token_urlsafe(32)
    return token, hash_key(token)


def _lookup(token: str | None) -> dict:
    if not token:
        raise HTTPException(401, "missing X-API-Key")
    rows = db.query(
        "SELECT id, name, scope, device_id FROM api_keys"
        " WHERE key_hash = %s AND revoked_at IS NULL",
        (hash_key(token),),
    )
    if not rows:
        raise HTTPException(401, "invalid API key")
    db.execute("UPDATE api_keys SET last_used_at = now() WHERE id = %s", (rows[0]["id"],))
    return rows[0]


def require_key(x_api_key: str | None = Header(None)) -> dict:
    return _lookup(x_api_key)


def require_admin(key: dict = Depends(require_key)) -> dict:
    if key["scope"] != "admin":
        raise HTTPException(403, "admin key required")
    return key


def require_ingest(key: dict = Depends(require_key)) -> dict:
    """Admin keys may also ingest — convenient for replaying a capture by hand."""
    if key["scope"] not in ("ingest", "admin"):
        raise HTTPException(403, "ingest key required")
    return key


# --- web UI session -------------------------------------------------------
# The admin UI can't send an X-API-Key header on a plain browser navigation, so
# the login form stores the admin key in an httponly cookie and every page
# re-verifies it against the hash table. No session table, no signing library:
# the cookie holds the real key and is checked the same way a header is.
# ponytail: raw key in an httponly cookie is fine on a single-user, Tailscale-
# only box; add signed sessions if this ever faces more than one operator.
COOKIE_NAME = "session_key"


def admin_from_cookie(token: str | None) -> dict | None:
    if not token:
        return None
    rows = db.query(
        "SELECT id, name, scope, device_id FROM api_keys"
        " WHERE key_hash = %s AND revoked_at IS NULL AND scope = 'admin'",
        (hash_key(token),),
    )
    return rows[0] if rows else None
