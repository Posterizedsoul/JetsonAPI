# Two hosts, one Dockerfile:
#   Jetson  : BASE_IMAGE=nvcr.io/nvidia/l4t-pytorch:...  (torch preinstalled)
#   dev/x86 : BASE_IMAGE=python:3.12-slim                (torch from PyPI)
ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE}

WORKDIR /app

COPY requirements.txt .
# The dustynv/L4T base ships an old pip and a pip.conf pointing at piwheels as
# the primary index, where "no versions found for fastapi" comes from. Upgrade
# pip and force PyPI as the index; keep piwheels as a fallback for arm64 binary
# wheels (pillow, psycopg-binary) that PyPI may not carry for aarch64.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        --index-url https://pypi.org/simple \
        --extra-index-url https://www.piwheels.org/simple \
        -r requirements.txt

# torch is not in requirements.txt because the L4T base image already ships a
# CUDA-enabled build that must not be overwritten by a PyPI wheel. Use python3,
# NOT python: the dustynv/L4T base has no `python` symlink, so `python` here
# silently fell through and installed CPU torch over the CUDA build. If the
# base genuinely has no torch (the x86 python:slim dev image), fall back to PyPI.
RUN python3 -c "import torch; print('base torch', torch.__version__)" \
    || pip install --no-cache-dir --index-url https://pypi.org/simple torch

COPY app ./app
COPY migrations ./migrations
# Shipped so the same checks can be run on the Jetson after deploy.
COPY tests ./tests
COPY scripts ./scripts

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
