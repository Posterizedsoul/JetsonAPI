#!/usr/bin/env bash
# End-to-end check: register both seed models, ingest a 3-view board, confirm
# a server prediction lands and agreement is computed.
set -euo pipefail

BASE="${BASE:-http://localhost:8000}"
ADMIN_KEY="${ADMIN_KEY:?set ADMIN_KEY}"
INGEST_KEY="${INGEST_KEY:?set INGEST_KEY}"
# Repo-relative on purpose: `docker compose cp` on Windows mangles absolute
# MSYS paths like /tmp/xxx into C:\tmp\xxx.
TMP=./.smoke-tmp
mkdir -p "$TMP"
trap 'rm -rf "$TMP"' EXIT

say() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }

say "health"
curl -sf "$BASE/health"; echo

say "register the board classifier (active)"
curl -s -X POST "$BASE/v1/models" -H "X-API-Key: $ADMIN_KEY" \
  -F "archive=@data/models/seed_board_clf.ts.pt" \
  -F "model_id=seed_board_clf" -F "version=v1" -F "activate=true"; echo

say "register the detector — different task, classes, input size, no views"
curl -s -X POST "$BASE/v1/models" -H "X-API-Key: $ADMIN_KEY" \
  -F "archive=@data/models/seed_detector.ts.pt" \
  -F "model_id=seed_detector" -F "version=v1" -F "activate=true"; echo

say "list models"
curl -sf "$BASE/v1/models" -H "X-API-Key: $ADMIN_KEY"; echo

say "make three lossless test captures"
# Generated in the gateway container so the host needs no Python deps.
MSYS_NO_PATHCONV=1 docker compose exec -T gateway python - <<'PY'
import random
from PIL import Image
random.seed(0)
for name in ("two_lights", "light_a", "light_b"):
    img = Image.new("RGB", (900, 700))
    img.putdata([(random.randint(0, 255),) * 3 for _ in range(900 * 700)])
    img.save(f"/tmp/{name}.png")      # PNG: lossless, never re-encoded
print("wrote 3 PNGs")
PY
for n in two_lights light_a light_b; do
    MSYS_NO_PATHCONV=1 docker compose cp "gateway:/tmp/$n.png" "$TMP/$n.png" >/dev/null
done

say "ingest a 3-view board with an edge prediction"
curl -sf -X POST "$BASE/v1/boards" -H "X-API-Key: $INGEST_KEY" \
  -F "board_id=board-0001" \
  -F "task=classification" \
  -F "views=@$TMP/two_lights.png" -F "lighting=both" \
  -F "views=@$TMP/light_a.png"    -F "lighting=light_a" \
  -F "views=@$TMP/light_b.png"    -F "lighting=light_b" \
  -F 'edge_prediction={"label":"gradeB","probs":{"gradeA":0.2,"gradeB":0.7,"gradeC":0.1},"confidence":0.7,"model_id":"seed_board_clf","model_version":"edge-light"}' \
  -F 'meta={"app_version":"1.0.0","gps":{"lat":45.1,"lon":-93.2},"white_balance":{"temp_k":5200}}'
echo

say "re-upload the SAME board (idempotency: no duplicates)"
curl -sf -X POST "$BASE/v1/boards" -H "X-API-Key: $INGEST_KEY" \
  -F "board_id=board-0001" \
  -F "views=@$TMP/two_lights.png" -F "lighting=both"; echo

say "wait for background inference"
sleep 4

say "predictions for the board"
curl -sf "$BASE/v1/boards/board-0001" -H "X-API-Key: $ADMIN_KEY"; echo

printf '\n\033[1;32mSmoke test complete.\033[0m\n'
