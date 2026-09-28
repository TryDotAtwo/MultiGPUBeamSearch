CREATE TABLE submissions (
  submission_id TEXT PRIMARY KEY,
  idempotency_key TEXT NOT NULL UNIQUE,
  run_id TEXT NOT NULL,
  author_name TEXT NOT NULL,
  competition TEXT NOT NULL,
  puzzle_type TEXT NOT NULL,
  puzzle_id INTEGER NOT NULL,
  state TEXT NOT NULL CHECK(state IN (
    'received','queued','validating','validated','rejected','staged','published','retryable','dead_letter'
  )),
  raw_r2_key TEXT NOT NULL UNIQUE,
  safe_error TEXT,
  retry_count INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  github_path TEXT,
  github_commit_sha TEXT
);
CREATE INDEX submissions_lookup ON submissions(competition,puzzle_type,puzzle_id,created_at);
CREATE INDEX submissions_run ON submissions(run_id,created_at);
CREATE INDEX submissions_recovery ON submissions(state,updated_at);
