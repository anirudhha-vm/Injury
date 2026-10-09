-- YouthTwin research database (SQLite / DuckDB dialect). Mirrors src/youthtwin/io/schema.py.
-- Lengths in cm, mass in kg. IDs are pseudonymous UUIDs. Dates are ISO-8601 text.
-- The ID-to-identity map is NOT part of this database and lives outside the repository.

CREATE TABLE IF NOT EXISTS athletes (
    id                TEXT PRIMARY KEY,
    sex               TEXT NOT NULL CHECK (sex IN ('male', 'female')),
    dob_month         TEXT NOT NULL,            -- YYYY-MM only, never the full date of birth
    sport             TEXT NOT NULL,
    academy_id        TEXT NOT NULL,
    dominant_limb     TEXT NOT NULL CHECK (dominant_limb IN ('left', 'right')),
    mid_parent_height REAL,
    synthetic         INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS anthropometrics (
    athlete_id        TEXT NOT NULL REFERENCES athletes (id),
    date              TEXT NOT NULL,
    stature_1         REAL NOT NULL,
    stature_2         REAL NOT NULL,
    sitting_height_1  REAL NOT NULL,
    sitting_height_2  REAL NOT NULL,
    mass_1            REAL NOT NULL,
    mass_2            REAL NOT NULL,
    measurer_id       TEXT NOT NULL,
    synthetic         INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (athlete_id, date)
);

CREATE TABLE IF NOT EXISTS screening_sessions (
    id                TEXT PRIMARY KEY,
    athlete_id        TEXT NOT NULL REFERENCES athletes (id),
    date              TEXT NOT NULL,
    camera_setup      TEXT NOT NULL,
    synthetic         INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS trials (
    id                TEXT PRIMARY KEY,
    session_id        TEXT NOT NULL REFERENCES screening_sessions (id),
    task              TEXT NOT NULL,
    limb              TEXT NOT NULL CHECK (limb IN ('left', 'right', 'both')),
    trial_no          INTEGER NOT NULL,
    keypoints_uri     TEXT NOT NULL,            -- keypoints only, never raw video
    quality           REAL NOT NULL,
    synthetic         INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS task_features (
    athlete_id        TEXT NOT NULL REFERENCES athletes (id),
    session_id        TEXT NOT NULL REFERENCES screening_sessions (id),
    task              TEXT NOT NULL,
    trial             INTEGER,                  -- NULL = session-level summary
    limb              TEXT NOT NULL CHECK (limb IN ('left', 'right', 'both')),
    feature           TEXT NOT NULL,
    value             REAL NOT NULL,
    quality           REAL NOT NULL,
    synthetic         INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS exposure_weekly (
    athlete_id        TEXT NOT NULL REFERENCES athletes (id),
    week_start        TEXT NOT NULL,
    hours_training    REAL NOT NULL,
    hours_match       REAL NOT NULL,
    sessions          INTEGER NOT NULL,
    synthetic         INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (athlete_id, week_start)
);

CREATE TABLE IF NOT EXISTS ostrc_weekly (
    athlete_id        TEXT NOT NULL REFERENCES athletes (id),
    week_start        TEXT NOT NULL,
    q1                INTEGER NOT NULL,
    q2                INTEGER NOT NULL,
    q3                INTEGER NOT NULL,
    q4                INTEGER NOT NULL,
    severity          INTEGER NOT NULL,
    synthetic         INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (athlete_id, week_start)
);

CREATE TABLE IF NOT EXISTS injuries (
    id                TEXT PRIMARY KEY,
    athlete_id        TEXT NOT NULL REFERENCES athletes (id),
    onset_date        TEXT NOT NULL,
    osiics_code       TEXT NOT NULL,
    body_area         TEXT NOT NULL,
    tissue            TEXT NOT NULL,
    mechanism         TEXT NOT NULL,
    time_loss_days    INTEGER NOT NULL,
    outcome_group     TEXT NOT NULL CHECK (outcome_group IN
                        ('growth_related', 'non_contact_soft_tissue', 'other_non_contact',
                         'contact', 'other')),
    diagnosed_by      TEXT NOT NULL,            -- outcome labels come only from a diagnosis (N1)
    synthetic         INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS growth_posteriors (
    athlete_id        TEXT NOT NULL REFERENCES athletes (id),
    fit_date          TEXT NOT NULL,
    uri               TEXT NOT NULL,
    synthetic         INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (athlete_id, fit_date)
);
