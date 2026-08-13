CREATE TABLE IF NOT EXISTS experiment_results (
    run_id       UUID NOT NULL DEFAULT gen_random_uuid(),
    experiment   TEXT NOT NULL,
    check_name   TEXT NOT NULL,
    metric_value DOUBLE PRECISION,
    passed       BOOLEAN,
    details      JSONB,
    created_at   TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (run_id, check_name)
);
CREATE INDEX IF NOT EXISTS idx_experiment_results_exp
    ON experiment_results(experiment, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_scores_poem_side_version
    ON scores(poem_id, side, metric_version);
