# Planned refactor: from board-centric to a general vision server

**Status: not started. This is a design note, not a task in progress.**

Today the server is built around GibsonNet's domain: the unit is a *board*, its
images are *views*, each view carries a *lighting condition*, and the label is a
*grade*. That wording is baked into the API, the database, the UI, and the
tests. It works, and it's model-agnostic in *behavior* (a single-image model
already runs fine — a board with one view). But the *vocabulary* assumes wood.

The goal of this refactor is to make the server read and work as a **general
vision inference server**: upload image(s), get a prediction whose shape is
decided by the model. Nothing wood-specific, nothing GibsonNet-specific, in any
name a user or client developer sees.

## Scope and non-goals

**In scope:** any *vision* model — classification, detection, segmentation, and
whatever future task types get added. Input is always **images**. Output is a
prediction whose shape the task type defines.

**Explicitly out of scope (do not build for these):**
- Text / LLM serving. This is a vision server. No prompts, no tokens, no text
  in, no text out.
- Any non-image modality (audio, tabular, sensor streams).

Keeping the scope to vision is what keeps the abstraction small. The moment
"any model" is taken to include LLMs, this turns into a plugin framework and a
service mesh — the exact over-engineering the current design avoids.

## The one thing that must survive: multi-image samples

The single hard constraint. Some models consume **several images of the same
subject jointly** — GibsonNet grades 1–3 lighting views of a board through one
shared encoder and an attention pooler. That is the reason a container unit
exists at all instead of bare images.

So the refactor is **not** "flatten everything to one image per prediction."
It's "rename the wood-specific container to a neutral one that still holds
1..N images." A single-image model is just the N=1 case.

## Target vocabulary

| Today (wood-specific) | Target (neutral) | Notes |
|---|---|---|
| board | **sample** | a unit holding 1..N images; one prediction per sample |
| view | **image** | one uploaded picture |
| lighting (`both`/`light_a`/`light_b`) | **tag** (optional, free text) | a per-image label the client may send; the server never interprets it |
| grade | **label** / **class** | already comes from archive metadata, never hardcoded |
| `board_id` | **`sample_id`** | client-generated, stable, survives export |

The prediction envelope already generalizes — `task` decides the result shape
(classification / detection / segmentation are all built in), so "any
prediction based on the model" is **already** how the contract works. No change
needed there beyond the rename.

## What a substantial refactor touches

This is why it's "substantial" and deferred, not a rename-in-place:

1. **Database** — rename `boards`→`samples`, `views`→`images`, columns
   (`board`→`sample`, `lighting`→`tag`). A migration that renames tables and
   columns and preserves all data. Foreign keys, indexes, and the unique
   idempotency constraints move with them.
2. **API contract** — `/v1/boards`→`/v1/samples`, form fields
   `board_id`→`sample_id`, `lighting`→`tag`. **This breaks existing field
   clients** (the phone app, the Blackfly host), so it needs a version bump and
   a documented migration for them — the whole reason to do it deliberately,
   not casually. Consider serving both `/v1/boards` and `/v1/samples` for one
   release with the old paths deprecated.
3. **Gateway code** — `routes.py`, `web.py`, `metrics.py` variable names and
   queries. Mechanical but wide.
4. **Web UI** — templates and nav ("Boards"→"Samples"). The UI is already
   task-driven for *rendering*; only the words change.
5. **Tests** — `test_runner.py` and any contract tests rename with the API.
6. **Docs** — `README.md`, `API.md` (when written), `models/README.md`.
7. **Training-set export** (when built) — must still preserve the stable id
   through the round trip; see the split-stability note below.

## Constraints to carry through the refactor

- **Model knowledge stays in archive metadata only.** Classes, input size,
  preprocessing, task — all read from `metadata.json`. The rename must not
  introduce a single hardcoded class name or task assumption. The existing
  leak-guard test (`test_no_class_names_leak_into_server_code`) should keep
  passing, extended to the new names.
- **Stable sample-id hashing.** Whatever `sample_id` becomes, the train/val/test
  split hashes it, and corrections must not change it. The unapplied GibsonNet
  `data.py` fix (hash the leaf id, not the grade-qualified path) still applies —
  see [gibsonnet-split-fix in memory / README "Known issues"].
- **One model resident at a time.** Unchanged; the 8GB limit doesn't care what
  the unit is called.
- **Multi-image is optional per model.** `max_views`/`multi_view` in metadata
  already drives this. A detector declares 1; GibsonNet declares 3. Keep that.

## Rough sequencing when it's time

1. Add `samples`/`images` as the new schema via migration, backfill from
   `boards`/`views`, keep the old tables as views or drop after cutover.
2. Add `/v1/samples` endpoints alongside `/v1/boards`; make the gateway write
   through to the new tables.
3. Move the UI and metrics to the new names.
4. Migrate the field clients to `/v1/samples`, then remove `/v1/boards`.
5. Rename internal variables and tests last, once the contract is settled.

Do it as one deliberate, versioned change with the field clients in the loop —
never a silent rename, because step 2 is a breaking API change.
