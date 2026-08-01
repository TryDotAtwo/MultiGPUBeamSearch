import { DurableObject } from "cloudflare:workers";
import { deleteStagedSubmission, findBySubmissionId, transition, type SubmissionRow } from "./db.js";
import { githubRequest, getInstallationToken, type GitHubAppConfig } from "./github-app.js";
import { canonicalJson, sha256Hex } from "./ids.js";
import { resolveIngestMode, type IngestMode } from "./mode.js";
import { MAX_SERIALIZED_ENVELOPE_BYTES, validateBatch, validateEnvelopeIntegrity, type ResultEnvelopeV1 } from "./schema.js";

export type WriterMode = IngestMode;
export const resolveWriterMode = resolveIngestMode;
export interface GitHubWriterEnv extends GitHubAppConfig { RESULTS_DB: D1Database; RAW_RESULTS: R2Bucket; INGEST_MODE?: string; STAGING_BRANCH?: string; }
export interface FlushResult { staged: number; retained: number; }
interface Pending { submissionId: string; }
interface Verified { id: string; row: SubmissionRow; envelope: ResultEnvelopeV1; body: string; path: string; }
interface Target { owner: string; repo: string; branchRoute: string; branchQuery: string; }
interface CommitResult { sha: string; staged: Verified[]; terminal: TerminalIntegrityError[]; }
const PENDING_PREFIX = "pending/", MAX_RECORDS = 100, MAX_BYTES = 5 * 1024 * 1024, MAX_DELAY_MS = 30_000, RETAIN_DELAY_MS = 15_000;
const SHA_RE = /^[0-9a-f]{40,64}$/;
const REPOSITORY_SEGMENT_RE = /^[A-Za-z0-9][A-Za-z0-9_.-]{0,99}$/;
const BRANCH_SEGMENT_RE = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/;
class TerminalIntegrityError extends Error { constructor(readonly submissionId: string, readonly safeError: "publication_raw_integrity_invalid" | "publication_path_conflict") { super(safeError); } }
function pendingKey(id: string): string { return `${PENDING_PREFIX}${id}`; }
function safeSegment(value: string): string { const out = value.toLowerCase().replace(/[^a-z0-9._-]+/g, "-").replace(/^-+|-+$/g, ""); if (!out || out.length > 128) throw new Error("github_path_invalid"); return out; }
function repositorySegment(value: string | undefined): string { const out = value?.trim(); if (!out || !REPOSITORY_SEGMENT_RE.test(out)) throw new Error("github_repository_invalid"); return out; }
export function branchRoute(value: string | undefined): string { const branch = value?.trim(); if (!branch || branch.length > 255 || branch.startsWith("/") || branch.endsWith("/")) throw new Error("github_branch_invalid"); const parts = branch.split("/"); if (parts.some((p) => p === "." || p === ".." || !BRANCH_SEGMENT_RE.test(p))) throw new Error("github_branch_invalid"); return parts.map(encodeURIComponent).join("/"); }
function target(env: GitHubWriterEnv): Target { const branch = env.STAGING_BRANCH?.trim(); const branchRouteValue = branchRoute(branch); if (!branch) throw new Error("github_branch_invalid"); return { owner: repositorySegment(env.REPO_OWNER), repo: repositorySegment(env.REPO_NAME), branchRoute: branchRouteValue, branchQuery: encodeURIComponent(branch) }; }
export function resultPath(id: string, record: ResultEnvelopeV1): string { const day = record.submitted_at.slice(0, 10); if (!/^\d{4}-\d{2}-\d{2}$/.test(day) || !/^[0-9a-f-]{36}$/.test(id)) throw new Error("github_path_invalid"); return ["results", "v1", safeSegment(record.competition), safeSegment(record.puzzle_type), String(record.puzzle_id), day, `${id}.json`].join("/"); }
function recordBody(id: string, envelope: ResultEnvelopeV1): string { return canonicalJson({ submission_id: id, envelope }); }
async function parseVerified(env: GitHubWriterEnv, id: string): Promise<Verified | null> {
  const row = await findBySubmissionId(env.RESULTS_DB, id);
  if (!row || row.state !== "validated") return null;
  const invalid = (): TerminalIntegrityError =>
    new TerminalIntegrityError(id, "publication_raw_integrity_invalid");
  const object = await env.RAW_RESULTS.get(row.raw_r2_key);
  if (object === null || object.size > MAX_SERIALIZED_ENVELOPE_BYTES) throw invalid();
  const raw = await object.text();
  if (new TextEncoder().encode(raw).byteLength > MAX_SERIALIZED_ENVELOPE_BYTES) throw invalid();
  const stored = object.customMetadata?.sha256;
  if (
    typeof stored !== "string" ||
    !/^[0-9a-f]{64}$/.test(stored) ||
    await sha256Hex(raw) !== stored
  ) {
    throw invalid();
  }
  let candidate: unknown;
  try {
    candidate = JSON.parse(raw);
  } catch {
    throw invalid();
  }
  const batch = validateBatch({ schema_version: 1, results: [candidate] });
  if (!batch.ok) throw invalid();
  const envelope = batch.value.results[0] as ResultEnvelopeV1;
  if (
    envelope.idempotency_key !== row.idempotency_key ||
    envelope.run_id !== row.run_id ||
    envelope.author.name !== row.author_name ||
    envelope.competition !== row.competition ||
    envelope.puzzle_type !== row.puzzle_type ||
    envelope.puzzle_id !== row.puzzle_id ||
    (await validateEnvelopeIntegrity(envelope)).length !== 0
  ) {
    throw invalid();
  }
  return {
    id,
    row,
    envelope,
    body: recordBody(id, envelope),
    path: resultPath(id, envelope),
  };
}
function sha(value: unknown): string | null { const candidate = value !== null && typeof value === "object" ? (value as Record<string, unknown>).sha : null; return typeof candidate === "string" && SHA_RE.test(candidate) ? candidate : null; }

function referenceSha(value: unknown): string | null {
  if (value === null || typeof value !== "object") return null;
  const record = value as Record<string, unknown>;
  return sha(record.object) ?? sha(record);
}
export class GitHubWriter extends DurableObject<GitHubWriterEnv> {
  private alarmUpdate: Promise<void> = Promise.resolve();
  constructor(ctx: DurableObjectState, env: GitHubWriterEnv) { super(ctx, env); ctx.blockConcurrencyWhile(async () => { const entries = await ctx.storage.list<Pending>({ prefix: PENDING_PREFIX, limit: 1 }); if (entries.size && await ctx.storage.getAlarm() === null) await this.rearm(); }); }
  private assertNormalMode(): void { if (resolveIngestMode(this.env.INGEST_MODE) !== "normal") throw new Error("writer_paused"); }
  private async guardedRequest(path: string, options: Parameters<typeof githubRequest>[2]) { this.assertNormalMode(); return githubRequest(this.env, path, options); }
  private async scheduleAlarmAt(deadline: number, force = false): Promise<void> { const update = this.alarmUpdate.then(async () => { const alarm = await this.ctx.storage.getAlarm(); if (force || alarm === null || alarm > deadline) await this.ctx.storage.setAlarm(deadline); }); this.alarmUpdate = update.catch(() => undefined); await update; }
  private async rearm(delayMs = RETAIN_DELAY_MS, force = false) { await this.scheduleAlarmAt(Date.now() + Math.min(MAX_DELAY_MS, Math.max(1, delayMs)), force); }
  async enqueueValidated(submissionId: string): Promise<void> { if (!/^[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/.test(submissionId)) throw new Error("writer_submission_invalid"); await this.ctx.storage.put(pendingKey(submissionId), { submissionId } satisfies Pending); await this.scheduleAlarmAt(Date.now() + RETAIN_DELAY_MS); }
  async alarm(): Promise<void> { await this.flush(true); }
  private async pending(): Promise<Pending[]> { const entries = await this.ctx.storage.list<Pending>({ prefix: PENDING_PREFIX, limit: MAX_RECORDS + 1 }); return [...entries.values()].slice(0, MAX_RECORDS); }
  private async terminalize(error: TerminalIntegrityError): Promise<void> { const changed = await transition(this.env.RESULTS_DB, error.submissionId, ["validated"], "dead_letter", { safeError: error.safeError }); if (!changed) { const current = await findBySubmissionId(this.env.RESULTS_DB, error.submissionId); if (!current || current.state !== "dead_letter" || current.safe_error !== error.safeError) throw new Error("writer_state_conflict"); } await this.ctx.storage.delete(pendingKey(error.submissionId)); }
  private async reconcile(value: Verified, commitSha: string): Promise<void> { const changed = await transition(this.env.RESULTS_DB, value.id, ["validated"], "staged", { safeError: null, githubPath: value.path, githubCommitSha: commitSha }); if (changed) return; const current = await findBySubmissionId(this.env.RESULTS_DB, value.id); if (!current || (current.state !== "staged" && current.state !== "published") || current.github_path !== value.path || !current.github_commit_sha || !SHA_RE.test(current.github_commit_sha)) throw new Error("writer_state_conflict"); }
  private async cleanupStaged(row: Pick<SubmissionRow, "submission_id" | "raw_r2_key" | "state">): Promise<void> { if (row.state !== "staged" && row.state !== "published") throw new Error("writer_state_conflict"); await this.env.RAW_RESULTS.delete(row.raw_r2_key); const deleted = await deleteStagedSubmission(this.env.RESULTS_DB, row.submission_id); if (!deleted) { const current = await findBySubmissionId(this.env.RESULTS_DB, row.submission_id); if (current) throw new Error("writer_state_conflict"); } await this.ctx.storage.delete(pendingKey(row.submission_id)); }
  async flush(forceFutureAlarm = false): Promise<FlushResult> { const pending = await this.pending(); if (!pending.length) return { staged: 0, retained: 0 }; if (resolveIngestMode(this.env.INGEST_MODE) !== "normal") { await this.rearm(RETAIN_DELAY_MS, forceFutureAlarm); return { staged: 0, retained: pending.length }; } const verified: Verified[] = []; let bytes = 0; try { for (const item of pending) { const current = await findBySubmissionId(this.env.RESULTS_DB, item.submissionId); if (current?.state === "staged" || current?.state === "published") { await this.cleanupStaged(current); continue; } let value: Verified | null; try { value = await parseVerified(this.env, item.submissionId); } catch (error) { if (error instanceof TerminalIntegrityError) { await this.terminalize(error); continue; } throw error; } if (!value) { await this.ctx.storage.delete(pendingKey(item.submissionId)); continue; } const size = new TextEncoder().encode(value.body).byteLength; if (size > MAX_BYTES) { if (!await transition(this.env.RESULTS_DB, value.id, ["validated"], "rejected", { safeError: "publication_record_oversize" })) throw new Error("writer_state_conflict"); await this.ctx.storage.delete(pendingKey(value.id)); continue; } if (verified.length && bytes + size > MAX_BYTES) break; verified.push(value); bytes += size; } if (!verified.length) { const retained = (await this.pending()).length; if (retained) await this.rearm(RETAIN_DELAY_MS, forceFutureAlarm); return { staged: 0, retained }; } const repo = target(this.env); this.assertNormalMode(); const token = await getInstallationToken(this.env, Date.now(), fetch, () => this.assertNormalMode()); const committed = await this.commit(token, repo, verified); for (const error of committed.terminal) await this.terminalize(error); for (const value of committed.staged) { await this.reconcile(value, committed.sha); const row = await findBySubmissionId(this.env.RESULTS_DB, value.id); if (!row) { await this.ctx.storage.delete(pendingKey(value.id)); continue; } await this.cleanupStaged(row); } const retained = (await this.pending()).length; if (retained) await this.rearm(RETAIN_DELAY_MS, forceFutureAlarm); return { staged: committed.staged.length, retained }; } catch { await this.rearm(RETAIN_DELAY_MS, forceFutureAlarm); throw new Error("github_writer_retryable"); } }
  private async preflight(token: string, repo: Target, item: Verified): Promise<"add" | "same"> { const route = item.path.split("/").map(encodeURIComponent).join("/"); const found = await this.guardedRequest(`/repos/${encodeURIComponent(repo.owner)}/${encodeURIComponent(repo.repo)}/contents/${route}?ref=${repo.branchQuery}`, { token }); if (found.status === 404) return "add"; const content = found.body !== null && typeof found.body === "object" ? (found.body as Record<string, unknown>).content : null; if (found.status !== 200 || typeof content !== "string") throw new Error("github_content_unavailable"); let decoded: string; try { decoded = new TextDecoder("utf-8", { fatal: true }).decode(Uint8Array.from(atob(content.replace(/\s/g, "")), (c) => c.charCodeAt(0))); } catch { throw new Error("github_content_unavailable"); } if (decoded !== item.body) throw new TerminalIntegrityError(item.id, "publication_path_conflict"); return "same"; }
  private async commit(token: string, repo: Target, items: Verified[]): Promise<CommitResult> { const terminal: TerminalIntegrityError[] = []; for (let retry = 0; retry < 3; retry += 1) { const ref = await this.guardedRequest(`/repos/${encodeURIComponent(repo.owner)}/${encodeURIComponent(repo.repo)}/git/ref/heads/${repo.branchRoute}`, { token }); const head = referenceSha(ref.body); if (ref.status !== 200 || !head) throw new Error("github_ref_unavailable"); const commit = await this.guardedRequest(`/repos/${encodeURIComponent(repo.owner)}/${encodeURIComponent(repo.repo)}/git/commits/${head}`, { token }); const baseTree = commit.body !== null && typeof commit.body === "object" ? sha((commit.body as Record<string, unknown>).tree) : null; if (commit.status !== 200 || !baseTree) throw new Error("github_commit_unavailable"); const additions: Verified[] = [], staged: Verified[] = []; for (const item of items) { if (terminal.some((entry) => entry.submissionId === item.id)) continue; try { if (await this.preflight(token, repo, item) === "add") additions.push(item); staged.push(item); } catch (error) { if (error instanceof TerminalIntegrityError) terminal.push(error); else throw error; } } if (!staged.length || !additions.length) return { sha: head, staged, terminal }; const tree = additions.map((item) => ({ path: item.path, mode: "100644", type: "blob", content: item.body })); const newTree = await this.guardedRequest(`/repos/${encodeURIComponent(repo.owner)}/${encodeURIComponent(repo.repo)}/git/trees`, { method: "POST", token, body: { base_tree: baseTree, tree } }); const treeSha = sha(newTree.body); if (newTree.status < 200 || newTree.status >= 300 || !treeSha) { if (newTree.status === 409 || newTree.status === 422) continue; throw new Error("github_tree_failed"); } const newCommit = await this.guardedRequest(`/repos/${encodeURIComponent(repo.owner)}/${encodeURIComponent(repo.repo)}/git/commits`, { method: "POST", token, body: { message: `ingest: stage ${additions.length} CayleyPy results`, tree: treeSha, parents: [head] } }); const commitSha = sha(newCommit.body); if (newCommit.status < 200 || newCommit.status >= 300 || !commitSha) { if (newCommit.status === 409 || newCommit.status === 422) continue; throw new Error("github_commit_failed"); } const update = await this.guardedRequest(`/repos/${encodeURIComponent(repo.owner)}/${encodeURIComponent(repo.repo)}/git/refs/heads/${repo.branchRoute}`, { method: "PATCH", token, body: { sha: commitSha, force: false } }); if (update.status >= 200 && update.status < 300) return { sha: commitSha, staged, terminal }; if (update.status !== 409 && update.status !== 422) throw new Error("github_ref_failed"); } throw new Error("github_ref_conflict"); }
}
