"""API key auth. Keys are stored as sha256 hashes; the plaintext is shown
once at creation and never again.

Ingest keys are bound to a device, so device_id is derived from the key rather
than trusted from the request body — a buggy client cannot attribute its
uploads to another device.
"""

import hashlib
import secrets
import time

from fastapi import Depends, Header, HTTPException

from app import db


def hash_key(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def new_key() -> tuple[str, str]:
    """(plaintext, hash). Show the plaintext once.

    16 chars, not 43: the box is only reachable over Tailscale, so ~96 bits is
    plenty and short enough to type on a borrowed machine. token_urlsafe(12)
    avoids +/ so it's terminal- and copy-friendly.
    ponytail: raise the byte count if this box is ever exposed beyond a tailnet.
    """
    token = secrets.token_urlsafe(12)
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


# --- one-time QR login tokens ---------------------------------------------
# So you can scan a QR on your phone instead of typing the admin key. A token
# carries the admin key's plaintext (handed in by the minting request, which
# already authenticated with it) for at most 5 minutes and a single use; the
# GET /ui/login?t= handler swaps it for the normal session cookie. The
# permanent key never travels in a URL, only this throwaway token does.
# ponytail: in-memory, single-gateway-process; a token dies on restart, which
# is fine — just reprint the QR from the access panel.
LOGIN_TOKEN_TTL = 300
_login_tokens: dict[str, tuple[str, float]] = {}


def mint_login_token(admin_key: str) -> str:
    now = time.time()
    # Opportunistic cleanup so the dict can't grow unbounded.
    for t, (_, exp) in list(_login_tokens.items()):
        if exp < now:
            _login_tokens.pop(t, None)
    token = secrets.token_urlsafe(9)
    _login_tokens[token] = (admin_key, now + LOGIN_TOKEN_TTL)
    return token


def consume_login_token(token: str) -> str | None:
    rec = _login_tokens.pop(token, None)
    if not rec:
        return None
    key, exp = rec
    return key if time.time() <= exp else None
