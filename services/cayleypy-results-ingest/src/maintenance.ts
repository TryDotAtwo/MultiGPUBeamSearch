import { findStagedSubmissions, findValidatedSubmissions } from "./db.js";
import { resolveIngestMode } from "./mode.js";
import { recoverStaleSubmissions, type IngestEnv } from "./storage.js";
import { RECOVERY_STALE_MS, RECOVERY_LIMIT } from "./ingest.js";

/** Runs only inside the publication Durable Object; the Cron edge delegates a timestamp. */
export async function maintainSubmissions(
  env: IngestEnv & { INGEST_MODE?: string },
  scheduledTime: number,
  enqueue: (id: string) => Promise<void>,
): Promise<void> {
  if (!Number.isSafeInteger(scheduledTime) || scheduledTime < 0 || scheduledTime > 8_640_000_000_000_000) throw new Error("maintenance_time_invalid");
  if (resolveIngestMode(env.INGEST_MODE) !== "normal") return;
  for (const row of await findStagedSubmissions(env.RESULTS_DB, 100)) {
    await env.RAW_RESULTS.delete(row.raw_r2_key);
  }
  for (const row of await findValidatedSubmissions(env.RESULTS_DB, 100)) {
    await enqueue(row.submission_id);
  }
  await recoverStaleSubmissions(env, {
    staleBefore: new Date(scheduledTime - RECOVERY_STALE_MS),
    limit: RECOVERY_LIMIT,
  });
}
