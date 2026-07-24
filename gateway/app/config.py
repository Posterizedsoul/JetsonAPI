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
