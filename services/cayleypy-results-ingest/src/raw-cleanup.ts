import { markRawDeleted, type SubmissionRow } from './db.js';

export async function cleanupRawPayload(
  db: D1Database,
  bucket: R2Bucket,
  row: Pick<SubmissionRow, 'submission_id' | 'state' | 'raw_r2_key'> & { raw_deleted_at?: string | null },
): Promise<void> {
  if (row.state !== 'staged' && row.state !== 'published') throw new Error('raw_cleanup_conflict');
  if (row.raw_deleted_at != null) return;
  await bucket.delete(row.raw_r2_key);
  await markRawDeleted(db, row.submission_id);
}
