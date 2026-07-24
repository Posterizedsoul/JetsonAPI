from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse

from app import db, routes, storage, web


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.pool.open(wait=True, timeout=30)
    db.migrate()
    storage.ensure_bucket()
    yield
    db.pool.close()


app = FastAPI(title="Jetson inference server", version="1", lifespan=lifespan)
app.include_router(routes.router)
app.include_router(web.router)


# UI dependencies raise _Redirect instead of a 401 so an unauthenticated
# browser is sent to the login page.
@app.exception_handler(web._Redirect)
def _redirect_handler(request: Request, exc: web._Redirect):
    return RedirectResponse(exc.url, status_code=303)


@app.get("/health")
def health() -> dict:
    checks = {}
    try:
        db.query("SELECT 1")
        checks["database"] = "ok"
    except Exception as exc:
        checks["database"] = f"error: {exc}"
    try:
        storage.client.head_bucket(Bucket=storage.config.S3_BUCKET)
        checks["storage"] = "ok"
    except Exception as exc:
        checks["storage"] = f"error: {exc}"
    ok = all(v == "ok" for v in checks.values())
    return {"status": "ok" if ok else "degraded", "checks": checks}
