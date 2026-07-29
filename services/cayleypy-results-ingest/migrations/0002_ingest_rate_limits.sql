CREATE TABLE ingest_rate_limits (
  scope TEXT PRIMARY KEY,
  window_start INTEGER NOT NULL,
  count INTEGER NOT NULL CHECK(count >= 0)
);
