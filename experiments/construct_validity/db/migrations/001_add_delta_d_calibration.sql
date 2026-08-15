CREATE EXTENSION IF NOT EXISTS "pgcrypto";

CREATE TABLE IF NOT EXISTS delta_d_calibration (
    construct_id    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    domain_cluster  INT NOT NULL,
    construct_type  TEXT NOT NULL CHECK (construct_type IN
                      ('probe_paraphrase','shuffle','pad','exact_h_member','ood')),
    source_poem_id  UUID REFERENCES poems(id),
    text            TEXT NOT NULL,
    n_symbols       INT,
    r_d_raw         DOUBLE PRECISION,
    r_d_norm        DOUBLE PRECISION,
    created_at      TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_delta_d_calib_domain
    ON delta_d_calibration(domain_cluster, construct_type);

CREATE TABLE IF NOT EXISTS delta_d_thresholds (
    domain_cluster  INT PRIMARY KEY,
    delta_d_95      DOUBLE PRECISION NOT NULL,
    n_neg           INT NOT NULL,
    quantile        DOUBLE PRECISION NOT NULL DEFAULT 0.95,
    created_at      TIMESTAMPTZ DEFAULT now()
);
