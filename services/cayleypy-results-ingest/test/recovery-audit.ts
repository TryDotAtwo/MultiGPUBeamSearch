import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

const RECEIPT_PREFIX = "CAYLEYPY_RECEIPT";
const MAX_MANIFEST_BYTES = 1_048_576;
const MAX_RECEIPT_EVENTS = 200;
const MAX_SNAPSHOT_BYTES = 4_194_304;
const MAX_SNAPSHOT_ROWS = 2_000;
const MAX_STATUS_BYTES = 16_384;
const MAX_ERRORS = 100;
const UUID_V7 = /^[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;
const SHA256 = /^[a-f0-9]{64}$/;

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
const SUBMISSION_STATES = new Set<string>(submissionStates);
const AUDIT_TERMINAL_STATES = new Set<SubmissionState>([
  "published",
  "rejected",
  "retryable",
  "dead_letter",
]);

export interface ReceiptManifestEntry {
  type: "receipt";
  workload_index: number;
  case_kind: "valid" | "duplicate";
  submission_id: string;
  idempotency_key: string;
  status_url: string;
}

export interface D1AuditRow {
  submission_id: string;
  idempotency_key: string;
  state: SubmissionState;
  raw_r2_key: string;
  safe_error: string | null;
  github_path: string | null;
}

export interface GitHubAuditEntry {
  path: string;
  submission_id: string;
  idempotency_key: string;
}

export interface SubmissionStatus {
  submission_id: string;
  idempotency_key: string;
  state: SubmissionState;
  safe_error: string | null;
  retry_count: number;
  updated_at: string;
}

export interface ExpectedWorkloadCounts {
  valid: number;
  duplicate: number;
  unique: number;
}

export interface RecoveryAuditInput {
  receipts: ReceiptManifestEntry[];
  d1Rows: D1AuditRow[];
  r2Keys: string[];
  githubEntries: GitHubAuditEntry[];
  statuses: SubmissionStatus[];
  requirePublished: boolean;
  expectedCounts?: ExpectedWorkloadCounts;
}

export interface RecoveryAuditResult {
  ok: boolean;
  errors: string[];
  summary: {
    receipt_events: number;
    unique_receipts: number;
    duplicate_events: number;
    published: number;
    rejected: number;
    recoverable: number;
    github_files: number;
  };
}

type UnknownRecord = Record<string, unknown>;

function byteLength(value: string): number {
  return new TextEncoder().encode(value).byteLength;
}

function isRecord(value: unknown): value is UnknownRecord {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function requiredString(record: UnknownRecord, key: string, code: string): string {
  const value = record[key];
  if (typeof value !== "string" || value.length === 0) throw new Error(code);
  return value;
}

function nullableString(record: UnknownRecord, key: string, code: string): string | null {
  const value = record[key];
  if (value === null) return null;
  if (typeof value !== "string" || value.length === 0) throw new Error(code);
  return value;
}

function parseJson(text: string, label: string): unknown {
  if (byteLength(text) > MAX_SNAPSHOT_BYTES) throw new Error(`${label}_too_large`);
  try {
    return JSON.parse(text) as unknown;
  } catch {
    throw new Error(`${label}_invalid_json`);
  }
}

function enforceRowLimit(rows: unknown[], label: string): void {
  if (rows.length > MAX_SNAPSHOT_ROWS) throw new Error(`${label}_row_limit`);
}

function parseReceipt(value: unknown): ReceiptManifestEntry {
  if (!isRecord(value) || value.type !== "receipt") throw new Error("manifest_invalid_receipt");
  const workloadIndex = value.workload_index;
  if (!Number.isInteger(workloadIndex) || (workloadIndex as number) < 0 || (workloadIndex as number) >= 100) {
    throw new Error("manifest_invalid_workload_index");
  }
  if (value.case_kind !== "valid" && value.case_kind !== "duplicate") {
    throw new Error("manifest_invalid_case_kind");
  }
  const submissionId = requiredString(value, "submission_id", "manifest_invalid_submission_id");
  const idempotencyKey = requiredString(value, "idempotency_key", "manifest_invalid_idempotency_key");
  const statusUrl = requiredString(value, "status_url", "manifest_invalid_status_url");
  if (!UUID_V7.test(submissionId)) throw new Error("manifest_invalid_submission_id");
  if (!SHA256.test(idempotencyKey)) throw new Error("manifest_invalid_idempotency_key");
  let parsedStatusUrl: URL;
  try {
    parsedStatusUrl = new URL(statusUrl);
  } catch {
    throw new Error("manifest_invalid_status_url");
  }
  if (
    !["http:", "https:"].includes(parsedStatusUrl.protocol)
    || parsedStatusUrl.pathname !== `/v1/submissions/${submissionId}`
  ) {
    throw new Error("manifest_invalid_status_url");
  }
  return {
    type: "receipt",
    workload_index: workloadIndex as number,
    case_kind: value.case_kind,
    submission_id: submissionId,
    idempotency_key: idempotencyKey,
    status_url: statusUrl,
  };
}

export function parseReceiptManifest(text: string): ReceiptManifestEntry[] {
  if (byteLength(text) > MAX_MANIFEST_BYTES) throw new Error("manifest_too_large");
  const receipts: ReceiptManifestEntry[] = [];
  for (const line of text.split(/\r?\n/)) {
    if (line.length === 0) continue;
    let message = line;
    if (line.startsWith("{")) {
      try {
        const parsed = JSON.parse(line) as unknown;
        if (isRecord(parsed) && typeof parsed.msg === "string") message = parsed.msg;
      } catch {
        // Non-JSON k6 output is intentionally ignored unless it carries the prefix.
      }
    }
    const prefix = `${RECEIPT_PREFIX}\t`;
    if (!message.startsWith(prefix)) continue;
    if (byteLength(message) > 8_192) throw new Error("manifest_line_too_large");
    let parsed: unknown;
    try {
      parsed = JSON.parse(message.slice(prefix.length)) as unknown;
    } catch {
      throw new Error("manifest_invalid_receipt_json");
    }
    receipts.push(parseReceipt(parsed));
    if (receipts.length > MAX_RECEIPT_EVENTS) throw new Error("manifest_receipt_limit");
  }
  if (receipts.length === 0) throw new Error("manifest_empty");
  return receipts;
}

function parseD1Row(value: unknown): D1AuditRow {
  if (!isRecord(value)) throw new Error("d1_invalid_row");
  const submissionId = requiredString(value, "submission_id", "d1_invalid_submission_id");
  const idempotencyKey = requiredString(value, "idempotency_key", "d1_invalid_idempotency_key");
  const state = requiredString(value, "state", "d1_invalid_state");
  if (!UUID_V7.test(submissionId)) throw new Error("d1_invalid_submission_id");
  if (!SHA256.test(idempotencyKey)) throw new Error("d1_invalid_idempotency_key");
  if (!SUBMISSION_STATES.has(state)) throw new Error("d1_invalid_state");
  return {
    submission_id: submissionId,
    idempotency_key: idempotencyKey,
    state: state as SubmissionState,
    raw_r2_key: requiredString(value, "raw_r2_key", "d1_invalid_raw_key"),
    safe_error: nullableString(value, "safe_error", "d1_invalid_safe_error"),
    github_path: nullableString(value, "github_path", "d1_invalid_github_path"),
  };
}

export function parseD1Snapshot(text: string): D1AuditRow[] {
  const parsed = parseJson(text, "d1_snapshot");
  if (!Array.isArray(parsed)) throw new Error("d1_snapshot_invalid_shape");
  const rows = parsed.length > 0 && isRecord(parsed[0]) && Array.isArray(parsed[0].results)
    ? parsed.flatMap((page) => {
        if (!isRecord(page) || !Array.isArray(page.results)) throw new Error("d1_snapshot_invalid_shape");
        return page.results;
      })
    : parsed;
  enforceRowLimit(rows, "d1_snapshot");
  return rows.map(parseD1Row);
}

export function parseR2Snapshot(text: string): string[] {
  const parsed = parseJson(text, "r2_snapshot");
  let objects: unknown[];
  if (Array.isArray(parsed)) {
    objects = parsed;
  } else if (isRecord(parsed) && Array.isArray(parsed.objects)) {
    objects = parsed.objects;
  } else {
    throw new Error("r2_snapshot_invalid_shape");
  }
  enforceRowLimit(objects, "r2_snapshot");
  return objects.map((value) => {
    if (typeof value === "string" && value.length > 0) return value;
    if (isRecord(value)) return requiredString(value, "key", "r2_snapshot_invalid_key");
    throw new Error("r2_snapshot_invalid_key");
  });
}

function parseGitHubEntry(value: unknown): GitHubAuditEntry {
  if (!isRecord(value)) throw new Error("github_snapshot_invalid_entry");
  const submissionId = requiredString(value, "submission_id", "github_snapshot_invalid_submission_id");
  const idempotencyKey = requiredString(value, "idempotency_key", "github_snapshot_invalid_idempotency_key");
  if (!UUID_V7.test(submissionId)) throw new Error("github_snapshot_invalid_submission_id");
  if (!SHA256.test(idempotencyKey)) throw new Error("github_snapshot_invalid_idempotency_key");
  return {
    path: requiredString(value, "path", "github_snapshot_invalid_path"),
    submission_id: submissionId,
    idempotency_key: idempotencyKey,
  };
}

export function parseGitHubSnapshot(text: string): GitHubAuditEntry[] {
  const parsed = parseJson(text, "github_snapshot");
  const entries = Array.isArray(parsed)
    ? parsed
    : isRecord(parsed) && Array.isArray(parsed.entries)
      ? parsed.entries
      : null;
  if (entries === null) throw new Error("github_snapshot_invalid_shape");
  enforceRowLimit(entries, "github_snapshot");
  return entries.map(parseGitHubEntry);
}

function parseStatus(value: unknown): SubmissionStatus {
  if (!isRecord(value)) throw new Error("status_invalid_shape");
  const submissionId = requiredString(value, "submission_id", "status_invalid_submission_id");
  const idempotencyKey = requiredString(value, "idempotency_key", "status_invalid_idempotency_key");
  const state = requiredString(value, "state", "status_invalid_state");
  if (!UUID_V7.test(submissionId)) throw new Error("status_invalid_submission_id");
  if (!SHA256.test(idempotencyKey)) throw new Error("status_invalid_idempotency_key");
  if (!SUBMISSION_STATES.has(state)) throw new Error("status_invalid_state");
  if (!Number.isInteger(value.retry_count) || (value.retry_count as number) < 0) {
    throw new Error("status_invalid_retry_count");
  }
  return {
    submission_id: submissionId,
    idempotency_key: idempotencyKey,
    state: state as SubmissionState,
    safe_error: nullableString(value, "safe_error", "status_invalid_safe_error"),
    retry_count: value.retry_count as number,
    updated_at: requiredString(value, "updated_at", "status_invalid_updated_at"),
  };
}

function mapUnique<T>(
  values: T[],
  key: (value: T) => string,
  duplicateCode: string,
  addError: (error: string) => void,
): Map<string, T> {
  const output = new Map<string, T>();
  for (const value of values) {
    const id = key(value);
    if (output.has(id)) addError(`${duplicateCode}:${id}`);
    else output.set(id, value);
  }
  return output;
}

export function auditRecoverySnapshot(input: RecoveryAuditInput): RecoveryAuditResult {
  const errors: string[] = [];
  const addError = (error: string) => {
    if (errors.length < MAX_ERRORS && !errors.includes(error)) errors.push(error);
  };

  if (input.receipts.length > MAX_RECEIPT_EVENTS) addError("manifest_receipt_limit");
  if (input.d1Rows.length > MAX_SNAPSHOT_ROWS) addError("d1_snapshot_row_limit");
  if (input.r2Keys.length > MAX_SNAPSHOT_ROWS) addError("r2_snapshot_row_limit");
  if (input.githubEntries.length > MAX_SNAPSHOT_ROWS) addError("github_snapshot_row_limit");
  if (input.statuses.length > MAX_RECEIPT_EVENTS) addError("status_row_limit");

  const receiptBySubmission = new Map<string, ReceiptManifestEntry>();
  const submissionByIdempotency = new Map<string, string>();
  const validKeys = new Set<string>();
  let duplicateEvents = 0;
  let validEvents = 0;
  for (const receipt of input.receipts.slice(0, MAX_RECEIPT_EVENTS)) {
    const priorSubmission = submissionByIdempotency.get(receipt.idempotency_key);
    if (priorSubmission !== undefined && priorSubmission !== receipt.submission_id) {
      addError(`receipt_idempotency_conflict:${receipt.idempotency_key}`);
    } else {
      submissionByIdempotency.set(receipt.idempotency_key, receipt.submission_id);
    }
    const priorReceipt = receiptBySubmission.get(receipt.submission_id);
    if (priorReceipt !== undefined && priorReceipt.idempotency_key !== receipt.idempotency_key) {
      addError(`receipt_submission_conflict:${receipt.submission_id}`);
    } else if (priorReceipt === undefined) {
      receiptBySubmission.set(receipt.submission_id, receipt);
    }
    if (receipt.case_kind === "duplicate") duplicateEvents += 1;
    else {
      validEvents += 1;
      validKeys.add(receipt.idempotency_key);
    }
  }
  for (const receipt of input.receipts) {
    if (receipt.case_kind === "duplicate" && !validKeys.has(receipt.idempotency_key)) {
      addError(`duplicate_without_unique:${receipt.idempotency_key}`);
    }
  }

  if (input.expectedCounts !== undefined) {
    if (validEvents !== input.expectedCounts.valid) {
      addError(`workload_valid_count:${validEvents}:${input.expectedCounts.valid}`);
    }
    if (duplicateEvents !== input.expectedCounts.duplicate) {
      addError(`workload_duplicate_count:${duplicateEvents}:${input.expectedCounts.duplicate}`);
    }
    if (receiptBySubmission.size !== input.expectedCounts.unique) {
      addError(`workload_unique_count:${receiptBySubmission.size}:${input.expectedCounts.unique}`);
    }
  }

  const d1BySubmission = mapUnique(
    input.d1Rows.slice(0, MAX_SNAPSHOT_ROWS),
    (row) => row.submission_id,
    "d1_duplicate_submission",
    addError,
  );
  const statusBySubmission = mapUnique(
    input.statuses.slice(0, MAX_RECEIPT_EVENTS),
    (current) => current.submission_id,
    "status_duplicate_submission",
    addError,
  );
  const rawKeys = new Set(input.r2Keys.slice(0, MAX_SNAPSHOT_ROWS));
  const githubByIdempotency = new Map<string, GitHubAuditEntry[]>();
  const githubPaths = new Set<string>();
  for (const entry of input.githubEntries.slice(0, MAX_SNAPSHOT_ROWS)) {
    if (githubPaths.has(entry.path)) addError(`github_duplicate_path:${entry.path}`);
    githubPaths.add(entry.path);
    const entries = githubByIdempotency.get(entry.idempotency_key) ?? [];
    entries.push(entry);
    githubByIdempotency.set(entry.idempotency_key, entries);
  }
  for (const [idempotencyKey, entries] of githubByIdempotency) {
    if (entries.length > 1) addError(`github_duplicate_idempotency:${idempotencyKey}`);
  }

  let published = 0;
  let rejected = 0;
  let recoverable = 0;
  for (const receipt of receiptBySubmission.values()) {
    const row = d1BySubmission.get(receipt.submission_id);
    if (row === undefined) {
      addError(`accepted_missing_d1:${receipt.submission_id}`);
      continue;
    }
    if (row.idempotency_key !== receipt.idempotency_key) {
      addError(`d1_idempotency_mismatch:${receipt.submission_id}`);
    }
    if (!rawKeys.has(row.raw_r2_key)) addError(`accepted_missing_raw:${receipt.submission_id}`);

    const current = statusBySubmission.get(receipt.submission_id);
    if (current === undefined) {
      addError(`accepted_missing_status:${receipt.submission_id}`);
    } else {
      if (current.idempotency_key !== receipt.idempotency_key) {
        addError(`status_idempotency_mismatch:${receipt.submission_id}`);
      }
      if (current.state !== row.state) {
        addError(`status_d1_state_mismatch:${receipt.submission_id}:${current.state}:${row.state}`);
      }
    }

    if (!AUDIT_TERMINAL_STATES.has(row.state)) {
      addError(`accepted_nonterminal_state:${receipt.submission_id}:${row.state}`);
      continue;
    }
    if (input.requirePublished && row.state !== "published") {
      addError(`validated_not_published:${receipt.submission_id}:${row.state}`);
    }

    if (row.state === "published") {
      published += 1;
      if (row.github_path === null) addError(`published_missing_github_path:${receipt.submission_id}`);
      const entries = githubByIdempotency.get(receipt.idempotency_key) ?? [];
      if (entries.length !== 1) {
        addError(`published_github_file_count:${receipt.submission_id}:${entries.length}`);
      } else {
        const entry = entries[0];
        if (entry.submission_id !== receipt.submission_id) {
          addError(`github_submission_mismatch:${receipt.submission_id}`);
        }
        if (row.github_path !== entry.path) addError(`github_path_mismatch:${receipt.submission_id}`);
      }
    } else if (row.state === "rejected") {
      rejected += 1;
      if (row.safe_error === null || row.safe_error.length === 0) {
        addError(`rejected_missing_reason:${receipt.submission_id}`);
      }
    } else {
      recoverable += 1;
      if (row.safe_error === null || row.safe_error.length === 0) {
        addError(`recoverable_missing_reason:${receipt.submission_id}`);
      }
    }
  }

  return {
    ok: errors.length === 0,
    errors,
    summary: {
      receipt_events: input.receipts.length,
      unique_receipts: receiptBySubmission.size,
      duplicate_events: duplicateEvents,
      published,
      rejected,
      recoverable,
      github_files: input.githubEntries.length,
    },
  };
}

export interface PollOptions {
  requirePublished: boolean;
  timeoutMs?: number;
  pollMs?: number;
  fetchImpl?: typeof fetch;
  sleepImpl?: (milliseconds: number) => Promise<void>;
}

function boundedInteger(value: number | undefined, fallback: number, minimum: number, maximum: number): number {
  if (value === undefined || !Number.isInteger(value)) return fallback;
  return Math.max(minimum, Math.min(maximum, value));
}

export async function pollReceiptStatuses(
  receipts: ReceiptManifestEntry[],
  endpointBase: string,
  options: PollOptions,
): Promise<SubmissionStatus[]> {
  const base = new URL(endpointBase);
  if (!["http:", "https:"].includes(base.protocol)) throw new Error("status_endpoint_invalid");
  if (receipts.length === 0 || receipts.length > MAX_RECEIPT_EVENTS) {
    throw new Error("status_receipt_limit");
  }
  const timeoutMs = boundedInteger(options.timeoutMs, 600_000, 0, 3_600_000);
  const pollMs = boundedInteger(options.pollMs, 5_000, 10, 60_000);
  const fetchImpl = options.fetchImpl ?? fetch;
  const sleepImpl = options.sleepImpl ?? ((milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds)));
  const unique = Array.from(new Map(receipts.map((entry) => [entry.submission_id, entry])).values());
  const statuses = new Map<string, SubmissionStatus>();
  const started = Date.now();

  while (true) {
    await Promise.all(unique.map(async (receipt) => {
      const url = new URL(`/v1/submissions/${receipt.submission_id}`, base);
      let response: Response;
      try {
        response = await fetchImpl(url, {
          method: "GET",
          headers: { Accept: "application/json" },
          redirect: "error",
        });
      } catch {
        return;
      }
      if (response.status !== 200) return;
      const text = await response.text();
      if (byteLength(text) > MAX_STATUS_BYTES) throw new Error("status_response_too_large");
      let parsed: unknown;
      try {
        parsed = JSON.parse(text) as unknown;
      } catch {
        throw new Error("status_invalid_json");
      }
      const current = parseStatus(parsed);
      if (current.submission_id !== receipt.submission_id) throw new Error("status_submission_mismatch");
      statuses.set(receipt.submission_id, current);
    }));

    const complete = unique.every((receipt) => {
      const current = statuses.get(receipt.submission_id);
      if (current === undefined) return false;
      if (!options.requirePublished) return AUDIT_TERMINAL_STATES.has(current.state);
      return current.state === "published" || current.state === "rejected" || current.state === "dead_letter";
    });
    if (complete) return unique.map((receipt) => statuses.get(receipt.submission_id)!);
    if (Date.now() - started >= timeoutMs) throw new Error("status_poll_timeout");
    await sleepImpl(pollMs);
  }
}

interface NodeProcess {
  argv: string[];
  env: Record<string, string | undefined>;
  exitCode?: number;
}

interface CliArguments {
  manifest: string;
  d1: string;
  r2: string;
  github: string;
  allowRecoverable: boolean;
}

function parseCliArguments(argv: string[]): CliArguments {
  const values = new Map<string, string>();
  let allowRecoverable = false;
  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index];
    if (argument === "--allow-recoverable") {
      allowRecoverable = true;
      continue;
    }
    if (!["--manifest", "--d1", "--r2", "--github"].includes(argument)) {
      throw new Error("invalid_argument");
    }
    const value = argv[index + 1];
    if (value === undefined || value.startsWith("--")) throw new Error("missing_argument_value");
    values.set(argument, value);
    index += 1;
  }
  const required = (name: string): string => {
    const value = values.get(name);
    if (value === undefined) throw new Error(`missing_argument:${name}`);
    return value;
  };
  return {
    manifest: required("--manifest"),
    d1: required("--d1"),
    r2: required("--r2"),
    github: required("--github"),
    allowRecoverable,
  };
}

function envMilliseconds(
  env: Record<string, string | undefined>,
  name: string,
  fallbackSeconds: number,
): number {
  const raw = env[name];
  if (raw === undefined) return fallbackSeconds * 1_000;
  const seconds = Number.parseInt(raw, 10);
  if (!Number.isInteger(seconds) || seconds < 0) throw new Error(`invalid_environment:${name}`);
  return seconds * 1_000;
}

async function main(nodeProcess: NodeProcess): Promise<void> {
  const args = parseCliArguments(nodeProcess.argv.slice(2));
  const endpointBase = nodeProcess.env.INGEST_BASE_URL;
  if (endpointBase === undefined || endpointBase.length === 0) {
    throw new Error("missing_environment:INGEST_BASE_URL");
  }
  const receipts = parseReceiptManifest(readFileSync(args.manifest, "utf8"));
  const requirePublished = !args.allowRecoverable;
  const statuses = await pollReceiptStatuses(receipts, endpointBase, {
    requirePublished,
    timeoutMs: envMilliseconds(nodeProcess.env, "RECOVERY_TIMEOUT_SECONDS", 600),
    pollMs: envMilliseconds(nodeProcess.env, "RECOVERY_POLL_SECONDS", 5),
  });
  const result = auditRecoverySnapshot({
    receipts,
    d1Rows: parseD1Snapshot(readFileSync(args.d1, "utf8")),
    r2Keys: parseR2Snapshot(readFileSync(args.r2, "utf8")),
    githubEntries: parseGitHubSnapshot(readFileSync(args.github, "utf8")),
    statuses,
    requirePublished,
    expectedCounts: { valid: 80, duplicate: 10, unique: 80 },
  });
  if (!result.ok) {
    console.error(JSON.stringify({ audit: "failed", errors: result.errors.slice(0, 20), summary: result.summary }));
    nodeProcess.exitCode = 1;
    return;
  }
  console.log(JSON.stringify({ audit: "passed", summary: result.summary }));
}

function normalizedPath(value: string): string {
  return value.replace(/\\/g, "/").toLowerCase();
}

const nodeProcess = (globalThis as unknown as { process?: NodeProcess }).process;
if (
  nodeProcess !== undefined
  && nodeProcess.argv[1] !== undefined
  && normalizedPath(nodeProcess.argv[1]) === normalizedPath(fileURLToPath(import.meta.url))
) {
  main(nodeProcess).catch((error: unknown) => {
    const reason = error instanceof Error && /^[A-Za-z0-9_:.-]+$/.test(error.message)
      ? error.message
      : "recovery_audit_failed";
    console.error(JSON.stringify({ audit: "failed", reason }));
    nodeProcess.exitCode = 1;
  });
}
