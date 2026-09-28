export const submissionStates = [
  "received",
  "queued",
  "validating",
  "validated",
  "rejected",
  "staged",
  "published",
  "retryable",
  "dead_letter",
] as const;
export type SubmissionState = typeof submissionStates[number];

export interface SubmissionRow {
  submission_id: string;
  idempotency_key: string;
  run_id: string;
  author_name: string;
  competition: string;
  puzzle_type: string;
  puzzle_id: number;
  state: SubmissionState;
  raw_r2_key: string;
  safe_error: string | null;
  retry_count: number;
  updated_at: string;
  github_path: string | null;
  github_commit_sha: string | null;
}

export interface TransitionPatch {
  safeError?: string | null;
  githubPath?: string;
  githubCommitSha?: string;
  incrementRetryCount?: boolean;
}

export async function findByIdempotency(db: D1Database, key: string): Promise<SubmissionRow | null> {
  return db
    .prepare("SELECT submission_id, idempotency_key, run_id, author_name, competition, puzzle_type, puzzle_id, state, raw_r2_key, safe_error, retry_count, updated_at, github_path, github_commit_sha FROM submissions WHERE idempotency_key = ?")
    .bind(key)
    .first<SubmissionRow>();
}

export async function findBySubmissionId(db: D1Database, id: string): Promise<SubmissionRow | null> {
  return db
    .prepare("SELECT submission_id, idempotency_key, run_id, author_name, competition, puzzle_type, puzzle_id, state, raw_r2_key, safe_error, retry_count, updated_at, github_path, github_commit_sha FROM submissions WHERE submission_id = ?")
    .bind(id)
    .first<SubmissionRow>();
}

export async function findStaleRecoverable(
  db: D1Database,
  staleBefore: string,
  limit: number,
): Promise<SubmissionRow[]> {
  const result = await db
    .prepare(
      "SELECT submission_id, idempotency_key, run_id, author_name, competition, puzzle_type, puzzle_id, state, raw_r2_key, safe_error, retry_count, updated_at, github_path, github_commit_sha FROM submissions WHERE state IN (?,?,?,?) AND updated_at <= ? ORDER BY updated_at, submission_id LIMIT ?",
    )
    .bind("received", "queued", "retryable", "validating", staleBefore, limit)
    .all<SubmissionRow>();
  return result.results;
}

/** Bounded, oldest-first operator inventory. Dead letters are never auto-replayed. */
export async function findDeadLetters(db: D1Database, limit: number): Promise<SubmissionRow[]> {
  const result = await db
    .prepare(
      "SELECT submission_id, idempotency_key, run_id, author_name, competition, puzzle_type, puzzle_id, state, raw_r2_key, safe_error, retry_count, updated_at, github_path, github_commit_sha FROM submissions WHERE state = ? ORDER BY updated_at, submission_id LIMIT ?",
    )
    .bind("dead_letter", limit)
    .all<SubmissionRow>();
  return result.results;
}
export async function findValidatedSubmissions(db: D1Database, limit: number): Promise<SubmissionRow[]> {
  const result = await db
    .prepare("SELECT submission_id, idempotency_key, run_id, author_name, competition, puzzle_type, puzzle_id, state, raw_r2_key, safe_error, retry_count, updated_at, github_path, github_commit_sha FROM submissions WHERE state = ? ORDER BY updated_at, submission_id LIMIT ?")
    .bind("validated", limit)
    .all<SubmissionRow>();
  return result.results;
}

export async function findStagedSubmissions(db: D1Database, limit: number): Promise<SubmissionRow[]> {
  const result = await db
    .prepare("SELECT submission_id, idempotency_key, run_id, author_name, competition, puzzle_type, puzzle_id, state, raw_r2_key, safe_error, retry_count, updated_at, github_path, github_commit_sha FROM submissions WHERE state IN (?,?) ORDER BY updated_at, submission_id LIMIT ?")
    .bind("staged", "published", limit)
    .all<SubmissionRow>();
  return result.results;
}

export async function deleteStagedSubmission(db: D1Database, id: string): Promise<boolean> {
  const result = await db
    .prepare("DELETE FROM submissions WHERE submission_id = ? AND state IN (?,?)")
    .bind(id, "staged", "published")
    .run();
  return result.meta.changes === 1;
}

export async function transition(
  db: D1Database,
  id: string,
  from: SubmissionState[],
  to: SubmissionState,
  patch: TransitionPatch = {},
): Promise<boolean> {
  if (from.length === 0) return false;
  const now = new Date().toISOString();
  const assignments = ["state = ?", "updated_at = ?"];
  const bindings: unknown[] = [to, now];
  if (patch.safeError !== undefined) {
    assignments.push("safe_error = ?");
    bindings.push(patch.safeError);
  }
  if (patch.githubPath !== undefined) {
    assignments.push("github_path = ?");
    bindings.push(patch.githubPath);
  }
  if (patch.githubCommitSha !== undefined) {
    assignments.push("github_commit_sha = ?");
    bindings.push(patch.githubCommitSha);
  }
  if (patch.incrementRetryCount) assignments.push("retry_count = retry_count + 1");
  bindings.push(id, ...from);
  const result = await db
    .prepare(
      `UPDATE submissions SET ${assignments.join(", ")} WHERE submission_id = ? AND state IN (${from.map(() => "?").join(",")})`,
    )
    .bind(...bindings)
    .run();
  return result.meta.changes === 1;
}
