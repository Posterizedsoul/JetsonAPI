# Jetson inference server

An always-on verification server for field-captured board images. Field clients
(an Android app, a FLIR Blackfly rig) capture a board, run a small model on
device, queue the result offline, and upload when they get connectivity. This
server stores the upload, runs the heavy model, and keeps both predictions so
you can measure how well the edge model is doing.

The server is **model-agnostic by construction**. Everything model-specific —
class names, input size, normalization, patch layout, calibration temperature —
lives in `metadata.json` embedded inside the TorchScript archive. Registering a
different model is an HTTP upload, not a code change.

---

## Status

Built and verified end to end:

| Capability | State |
|---|---|
| `docker compose up` → Postgres + MinIO + gateway, `/health` | working |
| Board-centric schema + migrations at boot | working |
| `ModelRunner` — load / unload / predict, one model resident | working |
| Two backends — TorchScript and ONNX Runtime, chosen by file extension | working |
| Model registry — register, list, activate, load, unload over HTTP | working |
| Ingest — multipart, 1–3 views, idempotent, lossless originals | working |
| Inference on ingest, edge + server predictions, agreement scoring | working |
| Review read API — sort by ordinal error, margin, confidence | working |
| Per-device API keys, sha256-hashed | working |
| Admin UI — dashboard, model register/activate/load/unload, key create/revoke | working |
| Performance page — live CPU/GPU/mem/temps + per-model latency & throughput | working |
| Running on the real Orin Nano, on GPU | working |
| `setup-jetson.sh` | deploys on hardware; see notes |

Admin UI is at `/ui` (log in with an admin key). Not built yet: the review
queue UI (image viewer + grading), replay, training-set export, verification
write endpoint, devices/records pages, per-model accuracy, `API.md`, `SETUP.md`,
`models/README.md`, and the benchmark script.

---

## Quick start

### On the Jetson

```bash
git clone <this repo> && cd JetsonAPI
sudo ./setup-jetson.sh --dry-run     # read what it will do first
sudo ./setup-jetson.sh               # base -> jetpack -> docker -> deploy
```

It ends by printing whether the gateway is healthy and where the admin key was
saved (`/mnt/nvme/jetsonapi/.admin-key-created`).

Individual steps, all idempotent:

```
preflight base jetpack power headless nvme swap docker tailscale deploy verify
```

Destructive, excluded from the default run, each confirms first:

```
nvme-format   wipe and repartition /dev/nvme0n1
nvme-boot     copy rootfs to NVMe and repoint the bootloader
```

### On a dev machine (x86, CPU)

```bash
cp .env.example .env                 # defaults are CPU/x86 already
docker compose up -d --build
curl localhost:8000/health

docker compose exec gateway python3 /app/scripts/make_seed_models.py --out /models
docker compose exec gateway python3 /app/scripts/create_key.py admin laptop
docker compose exec gateway python3 /app/scripts/create_key.py ingest phone-01 --device phone-01

ADMIN_KEY=... INGEST_KEY=... bash scripts/smoke.sh
```

---

## Layout

```
docker-compose.yml         postgres + minio + gateway
setup-jetson.sh            bring-up, deploy, verify
gateway/
  Dockerfile               one file, two hosts (L4T base or python:slim)
  app/
    main.py                app, lifespan, /health
    config.py              env vars, all with defaults
    db.py                  pool + migration runner
    storage.py             MinIO; originals untouched, previews separate
    auth.py                API keys, sha256-hashed
    runner.py              ModelRunner + per-task postprocess
    routes.py              /v1 endpoints
  migrations/001_init.sql  schema
  tests/test_runner.py     contract + agnosticism tests
  scripts/                 create_key.py, make_seed_models.py
scripts/
  smoke.sh                 end-to-end check
  test-setup-dryrun.sh     control-flow test for setup-jetson.sh
```

Ports: gateway on `0.0.0.0:8000` (reachable over Tailscale). Postgres and MinIO
bind `127.0.0.1` only. **Do not forward ports on your router** — Tailscale is
the only intended remote path.

---

## The API

All endpoints require `X-API-Key`. Admin keys are unbound; ingest keys are bound
to a device, and `device_id` is derived from the key rather than trusted from the
request body.

| Method | Path | Scope | Purpose |
|---|---|---|---|
| GET | `/health` | none | database + storage reachability |
| POST | `/v1/models` | admin | register an archive (multipart) |
| GET | `/v1/models` | any | list registered versions |
| POST | `/v1/models/{uuid}/activate` | admin | make it the active model for its task |
| POST | `/v1/models/{uuid}/load` | admin | load into memory |
| POST | `/v1/models/{uuid}/unload` | admin | free memory |
| POST | `/v1/boards` | ingest | upload a board and its views |
| GET | `/v1/boards` | any | review queue, filtered and sorted |
| GET | `/v1/boards/{board_id}` | any | one board with views and predictions |

### Ingest

```bash
curl -X POST http://jetson:8000/v1/boards \
  -H "X-API-Key: $INGEST_KEY" \
  -F "board_id=board-0001" \
  -F "task=classification" \
  -F "views=@two_lights.png" -F "lighting=both" \
  -F "views=@light_a.png"    -F "lighting=light_a" \
  -F "views=@light_b.png"    -F "lighting=light_b" \
  -F 'edge_prediction={"label":"gradeB","probs":{"gradeA":0.2,"gradeB":0.7,"gradeC":0.1},"confidence":0.7,"model_id":"gibsonnet","model_version":"light-v3"}' \
  -F 'meta={"app_version":"1.0.0","gps":{"lat":45.1,"lon":-93.2},"white_balance":{"temp_k":5200}}'
```

One `lighting` value per `views` file, in the same order. 1–3 views; partial
boards are normal, not an error.

**Idempotency.** Unique on `board_id`, and on `(board, sha256-of-bytes)` per
view. A client draining a queued upload after a failure gets the existing record
back with `duplicate: true` and no second inference. Retrying is always safe.

### Response envelope

The envelope is stable across model swaps. What changes between models is the
*content* of the prediction, never the shape:

```json
{
  "source": "server",
  "task": "classification",
  "label": "gradeA",
  "confidence": 0.4387,
  "margin": 0.064,
  "probs": {"gradeA": 0.4387, "gradeB": 0.1867, "gradeC": 0.3747},
  "view_attention": [0.3333, 0.3333, 0.3333],
  "outputs": {"calibrated": true, "patch_attention": [[...], [...], [...]]},
  "agrees": false,
  "ordinal_error": 1,
  "latency_ms": 944.18,
  "model_id": "gibsonnet", "model_version": "medium-v3"
}
```

`view_attention` is the pooler's per-view weight — which lighting condition drove
the grade. Keep it: it is the audit trail for a disputed call.

`ordinal_error` is the index distance between edge and server labels in the
model's own class list. The server knows an ordering exists, not what the grades
mean. Confusing adjacent grades is a much smaller error than skipping one, so
watch the mean of this alongside exact-match agreement.

---

## Registering a model

```bash
curl -X POST http://jetson:8000/v1/models \
  -H "X-API-Key: $ADMIN_KEY" \
  -F "archive=@gibsonnet_medium.ts.pt" \
  -F "model_id=gibsonnet" -F "version=medium-v3" \
  -F "activate=true"
```

Registration reads `metadata.json` out of the archive, validates it, and rejects
anything malformed with a 422 before the row is written.

### Manifest fields

Required — registration fails without them:

| Field | Meaning |
|---|---|
| `classes` | ordered list of class names. **The order is the ordinal order.** |
| `image_size` | model input side, px |
| `normalize_mean` / `normalize_std` | 3 floats each |

Optional, with defaults, so existing GibsonNet exports register unchanged:

| Field | Default | Meaning |
|---|---|---|
| `task` | `classification` | `classification` \| `detection` \| `segmentation` |
| `max_views` | `1` | views per sample |
| `patch_mode` | `false` | K native-resolution crops per view |
| `patches_per_view` | `4` | K |
| `temperature` | `1.0` | logits divided by this before softmax |
| `variant` | `null` | free label |
| `val_metrics` | `{}` | shown in the models list |

Derived if absent: `multi_view` (= `max_views > 1`) and `input_layout`
(`board` → `(B,V,C,H,W)` with a mask, `image` → `(B,C,H,W)`).

**Two fields are additive extensions to GibsonNet's current export format**:
`task` and `multi_view`. Archives that lack them load fine — absent `task` means
`classification`. Adding them to `gibsonnet/export.py` is worth doing so future
archives are explicit.

### Where model knowledge lives, exactly

- Class names → `metadata.json` only. A test asserts no class name appears in
  any `.py`, `.sql`, or `.html` under `app/` or `migrations/`. The seed models
  use deliberately fake names (`gradeA`, `knot`, …) so a leak from either
  archive shows up.
- Preprocessing → `runner.preprocess_view`, driven entirely by metadata.
- Class set size → `models.classes` (JSONB). A five-class model needs no
  migration.

**The one real boundary:** a new *task type* needs a postprocess function in
`runner.py` and a renderer in the UI. A new *model of an existing task* needs
neither. Classification, detection, and segmentation all exist already.

---

## Troubleshooting

### `docker compose up` — postgres reported unhealthy, gateway never started

First boot runs `initdb`, which is slow on a bind mount and can exceed the
healthcheck window. The compose file sets `start_period: 60s` for this. If it
still trips, just run `docker compose up -d` again — Postgres is healthy by then
and the gateway starts.

### Gateway is running on CPU when it should be on GPU

`deploy` checks this and warns. To confirm:

```bash
docker compose exec gateway python3 -c "import torch; print(torch.cuda.is_available())"
```

`False` means one of:

1. **The base image has no CUDA torch.** Check `GATEWAY_BASE_IMAGE` in `.env`.
   If it says `python:3.12-slim`, no L4T image could be pulled and it fell back
   to CPU. Log in to NGC (`docker login nvcr.io`) or pick a `dustynv/l4t-pytorch`
   tag matching your L4T version, then `docker compose up -d --build gateway`.
2. **nvidia is not the default docker runtime.** On Jetson, `--gpus all` does
   **not** work — the runtime must be the default so it can bind-mount L4T's
   CUDA libraries. Check with `docker info --format '{{.DefaultRuntime}}'`;
   it must print `nvidia`. Re-run `sudo ./setup-jetson.sh docker`.
3. **`nvidia-container-toolkit` is not installed.** Run the `jetpack` step first.

### `docker compose` — "compose is not a docker command"

`docker.io` from Ubuntu ships no Compose v2. `sudo ./setup-jetson.sh docker`
installs the arm64 plugin binary to
`/usr/local/lib/docker/cli-plugins/docker-compose`. Verify with
`docker compose version`.

### Out of memory / the box freezes during inference

8GB is unified CPU+GPU. Only one model is resident at a time by design.

- Confirm swap: `swapon --show`. The setup script puts 8GB on NVMe and disables
  zram (zram compresses into the RAM the model wants).
- Confirm headless: `systemctl get-default` should be `multi-user.target`.
  The desktop costs ~800MB.
- Unload before loading a different variant:
  `POST /v1/models/{uuid}/unload`.
- Patch mode multiplies compute by K per view. Three views × 4 patches × 4 TTA
  variants is 48 crops in one forward pass. Turn TTA off first if you are tight.
- Watch live: `jtop`.

### 409 "already registered"

`(model_id, version)` is unique. Bump the version — this is deliberate, so a
model version's prediction history can never be silently redefined.

### ONNX model won't register, or runs on CPU

Registration reads the manifest from inside the file. TorchScript archives from
`export.py` embed it; **ONNX files from `torch.onnx.export` do not**. Paste the
manifest into the **Metadata JSON** field on the Models page (or send a
`metadata` form field to `POST /v1/models`). It is stored with the model, so
loading works from then on. An embedded manifest always wins over the pasted one.

To embed it in the file instead, set `metadata_props["metadata.json"]` before
saving the ONNX model.

**On CPU?** The Performance page shows the execution provider under the GPU
gauge. `CPUExecutionProvider` means ONNX is not using the GPU — the PyPI
`onnxruntime` aarch64 wheel is **CPU-only**. For GPU ONNX on Jetson you need
NVIDIA's `onnxruntime-gpu` wheel (from `pypi.jetson-ai-lab.dev`) or a base image
that already ships one. The Dockerfile skips installing over an existing build
for exactly this reason. TorchScript models are unaffected.

### 422 on registration

The archive's `metadata.json` is missing, malformed, or missing a required key.
The message names the problem. Check what's actually embedded:

```bash
docker compose exec gateway python3 -c "
import torch, json
e = {'metadata.json': ''}
torch.jit.load('/models/your.ts.pt', map_location='cpu', _extra_files=e)
print(e['metadata.json'])"
```

### 401 / 403

401 = missing or unknown `X-API-Key`. 403 = wrong scope; model management needs
an admin key. Keys are stored hashed and cannot be recovered — mint a new one:

```bash
docker compose exec gateway python3 /app/scripts/create_key.py admin laptop
```

### Uploads succeed but no server prediction appears

Inference runs in the background after the response. Check, in order:

```bash
docker compose exec -T postgres psql -U gibson -d gibson \
  -c "SELECT kind, detail, created_at FROM errors ORDER BY created_at DESC LIMIT 10;"
```

The usual cause is `no active model for task <task>` — register a model with
`activate=true`, or activate an existing one. Note the `task` form field on
ingest must match a task that has an active model.

### `nvpmodel` picked the wrong mode

The script parses `/etc/nvpmodel.conf` and prefers a mode named `MAXN`, falling
back to the highest id. Override:

```bash
sudo nvpmodel -m <id>
sudo nvpmodel -q          # confirm
```

### Board came up unbootable after `nvme-boot`

The old boot device is untouched. Power off, remove the NVMe, boot as before,
then restore:

```bash
sudo cp /boot/extlinux/extlinux.conf.backup /boot/extlinux/extlinux.conf
```

This is why `nvme-boot` is excluded from the default run. Do it last, once
everything else works.

### Setup script died partway

Every step is idempotent — re-run it. Full log at `/var/log/jetson-setup.log`;
the error trap prints the failing line. To resume from a specific point:

```bash
sudo ./setup-jetson.sh docker tailscale deploy verify
```

If it says another run is in progress, check for a stale lock at
`/var/run/jetson-setup.lock`.

### Windows dev box: `docker compose cp` writes to `C:\tmp\...`

Git Bash rewrites MSYS absolute paths. Prefix commands with `MSYS_NO_PATHCONV=1`,
or use repo-relative paths — `scripts/smoke.sh` does the latter.

### Tests can't find the seed models

`tests/test_runner.py` skips if `/models/seed_board_clf.ts.pt` is absent. Build
them:

```bash
docker compose exec gateway python3 /app/scripts/make_seed_models.py --out /models
```

---

## Tests

```bash
docker compose exec gateway python3 -m pytest tests/ -q     # 17 tests
bash scripts/smoke.sh                                      # end to end
bash scripts/test-setup-dryrun.sh                          # setup script flow
```

The tests worth understanding:

- `test_no_class_names_leak_into_server_code` — the agnosticism guard. If this
  fails, something model-specific got hardcoded.
- `test_detector_loads_and_predicts` / `test_swapping_models_evicts_the_previous_one` —
  a second model with a different task, class count, input size, view count, and
  output shape, through the same code path. If adding a model ever requires
  editing `runner.py`, the abstraction is wrong; fix the abstraction rather than
  special-casing.
- `test_partial_boards` — one, two, and three views. The pooler is view-count
  agnostic and partial uploads must never be special-cased.
- `test_temperature_is_applied` — calibration is load-bearing, since the review
  queue ranks on confidence. A temperature that silently did nothing would
  corrupt the ordering without failing anything else.

---

## Known issues and decisions

**The split-stability bug in GibsonNet, and the agreed fix.** `data.py` builds
board ids as `f"{cls}/{entry.name}"` and `split_of()` hashes that string — so the
split key contains the label. A board corrected from one grade to another during
review exports under a different folder, gets a different sha1, and can cross the
train/val/test boundary. That silently contaminates exactly the comparison the
stable-hash design exists to protect, and the review queue is what triggers it.

Agreed fix (**not yet applied** — it touches the GibsonNet repo, not this one):
hash the leaf board name only, keeping the display id grade-qualified. Requires
board directory names to be unique across grades, which the export can
guarantee. One-line change, and it reassigns splits once on the next run.

**Attention is per-instance, reported per-view.** In patch mode the pooler
attends over V×K instances. `view_attention` sums each view's K patches, because
the question is "which lighting condition drove this", not "which patch".
`patch_attention` keeps the raw numbers.

**Padding removed.** GibsonNet's reference server padded every batch to
`max_views × K` and ran the encoder over zeros. Masked slots get exactly zero
attention weight, so dropping them is numerically identical and roughly 3×
cheaper on single-view boards.

**fp16 + `channels_last` on CUDA, fp32 on CPU.** Half-precision CPU kernels are
slower, not faster. The fp16-vs-fp32 accuracy delta needs the benchmark script
and real trained weights — untrained seed models would produce a meaningless
number.

**Not hardware-verified.** `setup-jetson.sh` passes shellcheck and a full
control-flow dry run against a faked Jetson environment, but has never run on an
Orin. The likeliest rough edges: the L4T PyTorch base image tag (it tries three
and falls back to CPU with a warning), `nvidia-container-toolkit` availability
in your L4T apt repo, and `nvme-boot`. Run `--dry-run` on the board first.
