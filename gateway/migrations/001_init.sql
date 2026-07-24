-- Board-centric schema. A board is the unit; views hang off it.
--
-- No grade name appears anywhere in this file, as a column, enum, or default.
-- Class sets live in models.classes, read from the archive's embedded
-- metadata. A model with five classes needs no migration.

CREATE TABLE devices (
    device_id          TEXT PRIMARY KEY,      -- client-generated, stable
    name               TEXT,
    app_version        TEXT,
    edge_model_id      TEXT,
    edge_model_version TEXT,
    last_seen_at       TIMESTAMPTZ,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Per-device keys: ingest requests derive device_id from the key, so a buggy
-- client cannot attribute uploads to someone else. scope='admin' keys are not
-- bound to a device.
CREATE TABLE api_keys (
    id           BIGSERIAL PRIMARY KEY,
    name         TEXT NOT NULL,
    key_hash     TEXT NOT NULL UNIQUE,        -- sha256 of the token
    scope        TEXT NOT NULL CHECK (scope IN ('admin', 'ingest')),
    device_id    TEXT REFERENCES devices(device_id),
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_used_at TIMESTAMPTZ,
    revoked_at   TIMESTAMPTZ,
    CHECK (scope <> 'ingest' OR device_id IS NOT NULL)
);

CREATE TABLE boards (
    id          UUID PRIMARY KEY,
    board_id    TEXT NOT NULL UNIQUE,         -- client-generated; survives export
    device_id   TEXT REFERENCES devices(device_id),
    captured_at TIMESTAMPTZ,
    received_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    flagged     BOOLEAN NOT NULL DEFAULT FALSE,
    -- GPS, white balance, colour reference card data, anything else the
    -- client sends. Stored verbatim, never interpreted on ingest.
    meta        JSONB NOT NULL DEFAULT '{}'::jsonb
);
CREATE INDEX boards_received_at_idx ON boards (received_at DESC);
CREATE INDEX boards_device_idx ON boards (device_id);

CREATE TABLE views (
    id          UUID PRIMARY KEY,
    board       UUID NOT NULL REFERENCES boards(id) ON DELETE CASCADE,
    -- 'both' | 'light_a' | 'light_b' as sent by the client. Free text on
    -- purpose: a future capture rig with different lighting must not need a
    -- migration. Never reordered, never dropped.
    lighting    TEXT NOT NULL,
    image_hash  TEXT NOT NULL,                -- sha256 of the original bytes
    object_key  TEXT NOT NULL,                -- lossless original in MinIO
    preview_key TEXT,                         -- downscaled JPEG, UI only
    width       INTEGER,
    height      INTEGER,
    format      TEXT,
    byte_size   BIGINT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- Idempotency: re-uploading a queued view is a no-op, not a duplicate.
    UNIQUE (board, image_hash)
);
CREATE INDEX views_board_idx ON views (board);

CREATE TABLE models (
    id            UUID PRIMARY KEY,
    model_id      TEXT NOT NULL,
    version       TEXT NOT NULL,
    task          TEXT NOT NULL,              -- from metadata; default classification
    archive_key   TEXT NOT NULL,
    -- The archive's embedded metadata.json, verbatim. This is the manifest.
    meta          JSONB NOT NULL,
    classes       JSONB NOT NULL,             -- ordered; order IS the ordinal order
    active        BOOLEAN NOT NULL DEFAULT FALSE,
    registered_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (model_id, version)
);
-- At most one active model per task.
CREATE UNIQUE INDEX models_one_active_per_task ON models (task) WHERE active;

CREATE TABLE replay_jobs (
    id          UUID PRIMARY KEY,
    model       UUID NOT NULL REFERENCES models(id),
    filter      JSONB NOT NULL DEFAULT '{}'::jsonb,
    status      TEXT NOT NULL DEFAULT 'pending'
                CHECK (status IN ('pending', 'running', 'done', 'failed', 'cancelled')),
    total       INTEGER NOT NULL DEFAULT 0,
    completed   INTEGER NOT NULL DEFAULT 0,
    error       TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at TIMESTAMPTZ
);

-- Append-only. Replay writes new rows; nothing here is ever updated or
-- deleted, so a model's history against real field data stays intact.
CREATE TABLE predictions (
    id             UUID PRIMARY KEY,
    board          UUID NOT NULL REFERENCES boards(id) ON DELETE CASCADE,
    source         TEXT NOT NULL CHECK (source IN ('edge', 'server')),
    model          UUID REFERENCES models(id),      -- null for edge predictions
    edge_model_id  TEXT,                            -- edge rows carry strings
    edge_model_ver TEXT,
    task           TEXT NOT NULL,
    -- Full calibrated distribution {class_name: p}, not just the argmax.
    probs          JSONB NOT NULL DEFAULT '{}'::jsonb,
    label          TEXT,                            -- argmax class name
    confidence     DOUBLE PRECISION,
    -- Gap between top two classes. Stored so the review queue can sort on it
    -- without unpacking JSON on every row.
    margin         DOUBLE PRECISION,
    -- Per-view pooling weights: which lighting condition drove the grade.
    view_attention JSONB,
    -- Per-patch outputs when patch mode is active; detection/segmentation
    -- results land here too, shaped by task.
    outputs        JSONB,
    -- Filled on server rows when an edge prediction exists to compare against.
    agrees         BOOLEAN,
    ordinal_error  INTEGER,
    latency_ms     DOUBLE PRECISION,
    tta            BOOLEAN NOT NULL DEFAULT FALSE,
    replay_job     UUID REFERENCES replay_jobs(id),
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX predictions_board_idx ON predictions (board);
CREATE INDEX predictions_model_idx ON predictions (model);
CREATE INDEX predictions_review_idx ON predictions (ordinal_error DESC NULLS LAST, margin ASC);
CREATE INDEX predictions_replay_idx ON predictions (replay_job);

-- Human labels are not predictions, so they do not live in that table.
-- Latest row per board wins; history is kept.
CREATE TABLE verifications (
    id          UUID PRIMARY KEY,
    board       UUID NOT NULL REFERENCES boards(id) ON DELETE CASCADE,
    label       TEXT NOT NULL,
    corrected   BOOLEAN NOT NULL DEFAULT FALSE,  -- true if it changed the server call
    note        TEXT,
    verified_by TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX verifications_board_idx ON verifications (board, created_at DESC);

CREATE TABLE errors (
    id         BIGSERIAL PRIMARY KEY,
    kind       TEXT NOT NULL,
    detail     TEXT NOT NULL,
    context    JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX errors_created_at_idx ON errors (created_at DESC);
