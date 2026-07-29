import { check, fail, sleep } from "k6";
import crypto from "k6/crypto";
import exec from "k6/execution";
import http from "k6/http";
import { Counter } from "k6/metrics";

const UNIQUE_VALID_RESULTS = 80;
const DUPLICATE_RESULTS = 10;
const INVALID_RESULTS = 10;
const TOTAL_RESULTS = UNIQUE_VALID_RESULTS + DUPLICATE_RESULTS + INVALID_RESULTS;
const MAX_429_RETRIES = 3;
const MAX_RETRY_AFTER_SECONDS = 60;
const RECEIPT_PREFIX = "CAYLEYPY_RECEIPT";

const acceptedReceipts = new Counter("accepted_receipts");
const duplicateReceipts = new Counter("duplicate_receipts");
const invalidRejections = new Counter("invalid_rejections");
const rateLimitRetries = new Counter("rate_limit_retries");
const unexpectedResponses = new Counter("unexpected_responses");

export const options = {
  vus: 100,
  iterations: 100,
  thresholds: {
    http_req_failed: ["rate<0.01"],
    http_req_duration: ["p(95)<2000"],
    checks: ["rate>0.99"],
  },
};

const endpointBase = (__ENV.INGEST_BASE_URL || "").replace(/\/+$/, "");
if (!/^https?:\/\/[^/]+(?:\/.*)?$/.test(endpointBase)) {
  throw new Error("INGEST_BASE_URL must be an http(s) origin or base URL");
}
const endpoint = `${endpointBase}/v1/results`;

const loadPhase = __ENV.LOAD_PHASE || "baseline";
if (!/^(baseline|github-outage|recovery)$/.test(loadPhase)) {
  throw new Error("LOAD_PHASE must be baseline, github-outage, or recovery");
}

const golden = JSON.parse(open("../../../configs/cayleypy_results_v1_golden.json"));
const baseCase = golden.cases.find((candidate) => candidate.name === "original_unicode_author");
if (!baseCase || !baseCase.envelope) {
  throw new Error("canonical golden original_unicode_author is missing");
}
const baseEnvelope = baseCase.envelope;

function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

function canonicalJson(value) {
  const seen = [];
  const encode = (item) => {
    if (item === null) return "null";
    switch (typeof item) {
      case "string":
        return JSON.stringify(item);
      case "boolean":
        return item ? "true" : "false";
      case "number":
        if (!Number.isFinite(item)) throw new Error("canonical_json_invalid_number");
        return Object.is(item, -0) ? "0" : String(item);
      case "object": {
        if (seen.indexOf(item) !== -1) throw new Error("canonical_json_cycle");
        seen.push(item);
        let output;
        if (Array.isArray(item)) {
          output = `[${item.map(encode).join(",")}]`;
        } else {
          output = `{${Object.keys(item)
            .sort()
            .map((key) => `${JSON.stringify(key)}:${encode(item[key])}`)
            .join(",")}}`;
        }
        seen.pop();
        return output;
      }
      default:
        throw new Error("canonical_json_unsupported_value");
    }
  };
  return encode(value);
}

function semanticEnvelope(envelope) {
  const semantic = clone(envelope);
  delete semantic.client_submission_id;
  delete semantic.run_id;
  delete semantic.idempotency_key;
  delete semantic.submitted_at;
  return semantic;
}

function computeIdempotency(envelope) {
  return crypto.sha256(canonicalJson(semanticEnvelope(envelope)), "hex");
}

function clientSubmissionId(transportIndex) {
  const tail = transportIndex.toString(16).padStart(12, "0");
  return `018f7a24-8f6b-7c8e-9d1b-${tail}`;
}

function applyTransport(envelope, transportIndex) {
  envelope.client_submission_id = clientSubmissionId(transportIndex + 1);
  envelope.run_id = `load-${loadPhase}-${transportIndex.toString().padStart(3, "0")}`;
  envelope.submitted_at = new Date(Date.UTC(2026, 6, 29, 12, 0, transportIndex)).toISOString();
}

function uniqueValidEnvelope(uniqueIndex) {
  const envelope = clone(baseEnvelope);
  const authorIndex = uniqueIndex % 20;
  envelope.author.name = `load-author-${authorIndex.toString().padStart(2, "0")}`;
  envelope.author.kaggle_username = `load-user-${authorIndex.toString().padStart(2, "0")}`;
  envelope.kaggle.owner = envelope.author.kaggle_username;
  envelope.kaggle.run_url = `https://www.kaggle.com/code/${envelope.kaggle.owner}/${envelope.kaggle.slug}`;
  envelope.puzzle_id = 10_000 + uniqueIndex;
  applyTransport(envelope, uniqueIndex);
  envelope.idempotency_key = computeIdempotency(envelope);
  return envelope;
}

function workloadCase(workloadIndex) {
  if (workloadIndex < UNIQUE_VALID_RESULTS) {
    return {
      kind: "valid",
      envelope: uniqueValidEnvelope(workloadIndex),
    };
  }

  if (workloadIndex < UNIQUE_VALID_RESULTS + DUPLICATE_RESULTS) {
    const duplicateIndex = workloadIndex - UNIQUE_VALID_RESULTS;
    const envelope = uniqueValidEnvelope(duplicateIndex);
    applyTransport(envelope, workloadIndex);
    return { kind: "duplicate", envelope };
  }

  const invalidIndex = workloadIndex - UNIQUE_VALID_RESULTS - DUPLICATE_RESULTS;
  const envelope = uniqueValidEnvelope(UNIQUE_VALID_RESULTS - INVALID_RESULTS + invalidIndex);
  envelope.proof.reached_state_sha256 = "0".repeat(64);
  applyTransport(envelope, workloadIndex);
  envelope.idempotency_key = computeIdempotency(envelope);
  return { kind: "invalid", envelope };
}

function retryAfterSeconds(response, workloadIndex, attempt) {
  const raw = response.headers["Retry-After"] || response.headers["retry-after"];
  const parsed = Number.parseInt(Array.isArray(raw) ? raw[0] : raw, 10);
  const base = Number.isFinite(parsed)
    ? Math.max(1, Math.min(MAX_RETRY_AFTER_SECONDS, parsed))
    : 1;
  const jitter = ((workloadIndex * 31 + attempt * 17) % 250) / 1000;
  return base + jitter;
}

function parseSafeJson(response) {
  try {
    return JSON.parse(response.body);
  } catch {
    return null;
  }
}

function responseParameters(kind) {
  const expected = kind === "invalid"
    ? http.expectedStatuses(400, 429)
    : kind === "duplicate"
      ? http.expectedStatuses(200, 202, 429)
      : http.expectedStatuses(202, 429);
  return {
    headers: {
      "Content-Type": "application/json",
      "User-Agent": "cayleypy-task8-k6",
    },
    responseCallback: expected,
    tags: { case_kind: kind, load_phase: loadPhase },
  };
}

function postWithBoundedRateLimitRetry(requestBody, workloadIndex, kind) {
  const parameters = responseParameters(kind);
  let response;
  for (let attempt = 0; attempt <= MAX_429_RETRIES; attempt += 1) {
    response = http.post(endpoint, requestBody, parameters);
    if (response.status !== 429) return response;

    const retryAfter = response.headers["Retry-After"] || response.headers["retry-after"];
    check(response, {
      "429 has bounded Retry-After": () => {
        const seconds = Number.parseInt(Array.isArray(retryAfter) ? retryAfter[0] : retryAfter, 10);
        return Number.isFinite(seconds) && seconds >= 1 && seconds <= MAX_RETRY_AFTER_SECONDS;
      },
    });
    if (attempt === MAX_429_RETRIES) return response;
    rateLimitRetries.add(1, { case_kind: kind, load_phase: loadPhase });
    sleep(retryAfterSeconds(response, workloadIndex, attempt));
  }
  return response;
}

function validateAcceptedResponse(response, workloadIndex, kind, expectedIdempotencyKey) {
  const parsed = parseSafeJson(response);
  const receipts = parsed && Array.isArray(parsed.receipts) ? parsed.receipts : [];
  const receipt = receipts.length === 1 ? receipts[0] : null;
  const ok = (
    receipt !== null
    && typeof receipt.submission_id === "string"
    && receipt.idempotency_key === expectedIdempotencyKey
    && typeof receipt.status_url === "string"
  );
  check(response, { "accepted response contains one matching safe receipt": () => ok });
  if (!ok) {
    unexpectedResponses.add(1, { case_kind: kind, load_phase: loadPhase });
    fail("accepted_response_contract_failed");
  }

  const manifestEntry = {
    type: "receipt",
    workload_index: workloadIndex,
    case_kind: kind,
    submission_id: receipt.submission_id,
    idempotency_key: receipt.idempotency_key,
    status_url: receipt.status_url,
  };
  console.log(`${RECEIPT_PREFIX}\t${JSON.stringify(manifestEntry)}`);
  acceptedReceipts.add(1, { case_kind: kind, load_phase: loadPhase });
  if (kind === "duplicate") duplicateReceipts.add(1, { load_phase: loadPhase });
}

export default function () {
  const workloadIndex = exec.scenario.iterationInTest;
  if (!Number.isInteger(workloadIndex) || workloadIndex < 0 || workloadIndex >= TOTAL_RESULTS) {
    fail("global_iteration_out_of_range");
  }

  sleep(((workloadIndex * 37) % 200) / 1000);
  const current = workloadCase(workloadIndex);
  const requestBody = JSON.stringify({ schema_version: 1, results: [current.envelope] });
  const response = postWithBoundedRateLimitRetry(
    requestBody,
    workloadIndex,
    current.kind,
  );

  if (response.status >= 500) {
    check(response, { "unexpected 5xx is forbidden": () => false });
    unexpectedResponses.add(1, { case_kind: current.kind, load_phase: loadPhase });
    fail("unexpected_server_error");
  }
  if (response.status === 429) {
    check(response, { "bounded 429 retries eventually complete": () => false });
    unexpectedResponses.add(1, { case_kind: current.kind, load_phase: loadPhase });
    fail("rate_limit_retry_exhausted");
  }

  if (current.kind === "invalid") {
    const parsed = parseSafeJson(response);
    const ok = response.status === 400 && parsed !== null && parsed.error === "invalid_schema";
    check(response, { "invalid proof is rejected without a receipt": () => ok });
    if (!ok) {
      unexpectedResponses.add(1, { case_kind: current.kind, load_phase: loadPhase });
      fail("invalid_proof_was_not_rejected");
    }
    invalidRejections.add(1, { load_phase: loadPhase });
    return;
  }

  const acceptedStatus = current.kind === "duplicate"
    ? response.status === 200 || response.status === 202
    : response.status === 202;
  check(response, { "valid result has accepted status": () => acceptedStatus });
  if (!acceptedStatus) {
    unexpectedResponses.add(1, { case_kind: current.kind, load_phase: loadPhase });
    fail("valid_result_not_accepted");
  }
  validateAcceptedResponse(response, workloadIndex, current.kind, current.envelope.idempotency_key);
}
