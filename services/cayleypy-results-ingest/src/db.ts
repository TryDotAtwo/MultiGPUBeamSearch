export const submissionStates = ["received", "queued", "validating", "validated", "rejected", "staged", "published", "retryable", "dead_letter"] as const;
export type SubmissionState = typeof submissionStates[number];

export interface SubmissionRow {
  submission_id: string; idempotency_key: string; state: SubmissionState; raw_r2_key: string;
}

export interface TransitionPatch { safeError?: string; githubPath?: string; githubCommitSha?: string }

export async function findByIdempotency(db: D1Database, key: string): Promise<SubmissionRow | null> {
  return db.prepare("SELECT submission_id, idempotency_key, state, raw_r2_key FROM submissions WHERE idempotency_key = ?").bind(key).first<SubmissionRow>();
}

export async function transition(db: D1Database, id: string, from: SubmissionState[], to: SubmissionState, patch: TransitionPatch = {}): Promise<boolean> {
  if (from.length === 0) return false;
  const now = new Date().toISOString();
  const assignments = ["state = ?", "updated_at = ?"];
  const bindings: unknown[] = [to, now];
  if (patch.safeError !== undefined) { assignments.push("safe_error = ?"); bindings.push(patch.safeError); }
  if (patch.githubPath !== undefined) { assignments.push("github_path = ?"); bindings.push(patch.githubPath); }
  if (patch.githubCommitSha !== undefined) { assignments.push("github_commit_sha = ?"); bindings.push(patch.githubCommitSha); }
  bindings.push(id, ...from);
  const result = await db.prepare(`UPDATE submissions SET ${assignments.join(", ")} WHERE submission_id = ? AND state IN (${from.map(() => "?").join(",")})`).bind(...bindings).run();
  return result.meta.changes === 1;
}
