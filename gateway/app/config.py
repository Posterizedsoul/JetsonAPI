import os

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://gibson:gibson@localhost:5432/gibson"
)
S3_ENDPOINT = os.environ.get("S3_ENDPOINT", "http://localhost:9000")
S3_ACCESS_KEY = os.environ.get("S3_ACCESS_KEY", "minioadmin")
S3_SECRET_KEY = os.environ.get("S3_SECRET_KEY", "minioadmin")
S3_BUCKET = os.environ.get("S3_BUCKET", "boards")
MODEL_DIR = os.environ.get("MODEL_DIR", "/models")

# "cuda", "cpu", or empty to auto-detect at load time.
DEVICE = os.environ.get("DEVICE", "")

# Long edge of the UI thumbnail. Previews are JPEG; originals never are.
PREVIEW_MAX_PX = int(os.environ.get("PREVIEW_MAX_PX", "1600"))

# How long an admin browser session stays valid. Default ~10 years: the box
# lives on a private tailnet, gets left unattended for weeks at a time, and
# being logged out while away from it is a worse outcome than a long cookie.
# Lower it if the server ever faces more than its operator.
SESSION_DAYS = int(os.environ.get("SESSION_DAYS", "3650"))
