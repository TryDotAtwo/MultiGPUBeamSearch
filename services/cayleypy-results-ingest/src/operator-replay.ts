import { findDeadLetters, transition } from "./db.js";

export interface OperatorReplayEnv {
  RESULTS_DB: D1Database;
  RAW_RESULTS: R2Bucket;
}

export interface OperatorReplayOptions {
  /** Explicit opt-in; omitting this is a non-mutating inspection. */
  apply?: true;
  /** Bounded operator page, clamped to 1..100. */
  limit?: number;
}

export interface OperatorReplayResult {
  dry_run: boolean;
  selected: string[];
  replayed: string[];
  skipped_missing_raw: string[];
}

const DEFAULT_LIMIT = 25;
const MAX_LIMIT = 100;

function boundedLimit(value: number | undefined): number {
  if (!Number.isInteger(value)) return DEFAULT_LIMIT;
  return Math.max(1, Math.min(value, MAX_LIMIT));
}

/**
 * Operator-only recovery primitive. It never runs from HTTP, Queue, or cron.
 * A row leaves dead_letter only with explicit apply=true and retained raw R2.
 */
export async function replayDeadLetters(
  env: OperatorReplayEnv,
  options: OperatorReplayOptions = {},
): Promise<OperatorReplayResult> {
  const rows = await findDeadLetters(env.RESULTS_DB, boundedLimit(options.limit));
  const selected = rows.map((row) => row.submission_id);
  if (options.apply !== true) {
    return { dry_run: true, selected, replayed: [], skipped_missing_raw: [] };
  }

  const replayed: string[] = [];
  const skippedMissingRaw: string[] = [];
  for (const row of rows) {
    const raw = await env.RAW_RESULTS.head(row.raw_r2_key);
    if (raw === null) {
      skippedMissingRaw.push(row.submission_id);
      continue;
    }
    if (await transition(env.RESULTS_DB, row.submission_id, ["dead_letter"], "retryable", {
      safeError: "operator_replay_pending",
    })) {
      replayed.push(row.submission_id);
    }
  }
  return { dry_run: false, selected, replayed, skipped_missing_raw: skippedMissingRaw };
}