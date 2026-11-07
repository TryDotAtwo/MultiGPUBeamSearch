import { findStagedSubmissions, findValidatedSubmissions } from "./db.js";
import { resolveIngestMode } from "./mode.js";
import { recoverStaleSubmissions, type IngestEnv } from "./storage.js";
import { RECOVERY_STALE_MS, RECOVERY_LIMIT } from "./ingest.js";
import { cleanupRawPayload } from "./raw-cleanup.js";

export const RAW_CLEANUP_INTERVAL_MS = 10 * 60_000;

/** Runs only inside the publication Durable Object; the Cron edge delegates a timestamp. */
export async function maintainSubmissions(
  env: IngestEnv & { INGEST_MODE?: string },
  scheduledTime: number,
  enqueue: (id: string) => Promise<void>,
): Promise<void> {
  if (!Number.isSafeInteger(scheduledTime) || scheduledTime < 0 || scheduledTime > 8_640_000_000_000_000) throw new Error("maintenance_time_invalid");
  if (resolveIngestMode(env.INGEST_MODE) !== "normal") return;
  // Recovery still runs each minute; only backup R2 cleanup runs every ten minutes.
  if (scheduledTime % RAW_CLEANUP_INTERVAL_MS < 60_000) {
    for (const row of await findStagedSubmissions(env.RESULTS_DB, 100)) {
      await cleanupRawPayload(env.RESULTS_DB, env.RAW_RESULTS, row);
    }
  }
  for (const row of await findValidatedSubmissions(env.RESULTS_DB, 100)) {
    await enqueue(row.submission_id);
  }
  await recoverStaleSubmissions(env, {
    staleBefore: new Date(scheduledTime - RECOVERY_STALE_MS),
    limit: RECOVERY_LIMIT,
  });
}
