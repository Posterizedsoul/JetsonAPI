# Two hosts, one Dockerfile:
#   Jetson  : BASE_IMAGE=nvcr.io/nvidia/l4t-pytorch:...  (torch preinstalled)
#   dev/x86 : BASE_IMAGE=python:3.12-slim                (torch from PyPI)
ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE}

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# torch is not in requirements.txt because the L4T base image already ships a
# CUDA-enabled build that must not be overwritten by a PyPI wheel.
RUN python -c "import torch" 2>/dev/null || pip install --no-cache-dir torch

COPY app ./app
COPY migrations ./migrations
# Shipped so the same checks can be run on the Jetson after deploy.
COPY tests ./tests
COPY scripts ./scripts

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
