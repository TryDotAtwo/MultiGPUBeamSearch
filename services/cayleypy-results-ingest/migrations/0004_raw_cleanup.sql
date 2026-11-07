-- Keep submission/idempotency evidence; remove completed R2 cleanup from the work queue.
ALTER TABLE submissions ADD COLUMN raw_deleted_at TEXT;
CREATE INDEX submissions_raw_cleanup
  ON submissions(updated_at, submission_id)
  WHERE raw_deleted_at IS NULL AND state IN ('staged', 'published');
