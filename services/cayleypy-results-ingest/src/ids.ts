import type { ResultEnvelopeV1 } from "./schema.js";

const encoder = new TextEncoder();

/** Stable JSON for hashing. It rejects values that JSON would silently change. */
export function canonicalJson(value: unknown): string {
  const seen = new Set<object>();
  const encode = (item: unknown): string => {
    if (item === null) return "null";
    switch (typeof item) {
      case "string": return JSON.stringify(item);
      case "boolean": return item ? "true" : "false";
      case "number":
        if (!Number.isFinite(item)) throw new TypeError("canonical_json_invalid_number");
        return Object.is(item, -0) ? "0" : String(item);
      case "object": {
        if (seen.has(item)) throw new TypeError("canonical_json_cycle");
        seen.add(item);
        let output: string;
        if (Array.isArray(item)) {
          output = `[${item.map(encode).join(",")}]`;
        } else {
          const record = item as Record<string, unknown>;
          output = `{${Object.keys(record).sort().map((key) => `${JSON.stringify(key)}:${encode(record[key])}`).join(",")}}`;
        }
        seen.delete(item);
        return output;
      }
      default: throw new TypeError("canonical_json_unsupported_value");
    }
  };
  return encode(value);
}

function semanticEnvelope(envelope: ResultEnvelopeV1): Omit<ResultEnvelopeV1, "submission_id" | "idempotency_key" | "submitted_at"> {
  const { submission_id: _submissionId, idempotency_key: _idempotencyKey, submitted_at: _submittedAt, ...semantic } = envelope;
  return semantic;
}

export async function sha256Hex(value: string): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", encoder.encode(value));
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join("");
}

/** Hashes the result meaning, not client-generated transport identifiers. */
export async function computeIdempotency(envelope: ResultEnvelopeV1): Promise<string> {
  return sha256Hex(canonicalJson(semanticEnvelope(envelope)));
}

export function newSubmissionId(now = new Date(), random = crypto.getRandomValues(new Uint8Array(10))): string {
  const bytes = new Uint8Array(16);
  let millis = now.getTime();
  for (let index = 5; index >= 0; index -= 1) { bytes[index] = millis & 0xff; millis = Math.floor(millis / 256); }
  bytes.set(random, 6);
  bytes[6] = (bytes[6] & 0x0f) | 0x70;
  bytes[8] = (bytes[8] & 0x3f) | 0x80;
  const hex = Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join("");
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
}

export function rawObjectKey(submissionId: string, now = new Date()): string {
  const yyyy = String(now.getUTCFullYear());
  const mm = String(now.getUTCMonth() + 1).padStart(2, "0");
  const dd = String(now.getUTCDate()).padStart(2, "0");
  return `raw/v1/${yyyy}/${mm}/${dd}/${submissionId}.json`;
}
