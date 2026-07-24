"""MinIO / S3 object storage.

Originals are stored as the client sent them, byte for byte. Previews are a
separate object and exist only so the UI has something small to render.
"""

import io

import boto3
from botocore.client import Config
from PIL import Image

from app import config

client = boto3.client(
    "s3",
    endpoint_url=config.S3_ENDPOINT,
    aws_access_key_id=config.S3_ACCESS_KEY,
    aws_secret_access_key=config.S3_SECRET_KEY,
    config=Config(signature_version="s3v4"),
    region_name="us-east-1",
)


def ensure_bucket() -> None:
    existing = {b["Name"] for b in client.list_buckets().get("Buckets", [])}
    if config.S3_BUCKET not in existing:
        client.create_bucket(Bucket=config.S3_BUCKET)


def put(key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
    client.put_object(
        Bucket=config.S3_BUCKET, Key=key, Body=data, ContentType=content_type
    )
    return key


def get(key: str) -> bytes:
    return client.get_object(Bucket=config.S3_BUCKET, Key=key)["Body"].read()


def make_preview(data: bytes, max_px: int | None = None) -> tuple[bytes, int, int, str]:
    """Downscaled JPEG for the UI, plus the ORIGINAL dimensions and format.

    The JPEG is a display artifact only. Inference always reads the original
    object: re-encoding grain boundaries produces artifacts indistinguishable
    from the fine curl detail the grade depends on.
    """
    max_px = max_px or config.PREVIEW_MAX_PX
    with Image.open(io.BytesIO(data)) as img:
        width, height, fmt = img.width, img.height, (img.format or "")
        preview = img.convert("RGB")
        preview.thumbnail((max_px, max_px), Image.BICUBIC)
        buf = io.BytesIO()
        # No colour correction, no ICC stripping beyond the resize: stain and
        # discolouration grading depends on tone being left alone.
        preview.save(buf, format="JPEG", quality=88)
    return buf.getvalue(), width, height, fmt
