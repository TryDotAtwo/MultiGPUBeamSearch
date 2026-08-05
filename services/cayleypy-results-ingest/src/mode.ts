export type IngestMode = "normal" | "store_only" | "reject";

export function resolveIngestMode(value: unknown): IngestMode {
  return value === "normal" || value === "store_only" || value === "reject" ? value : "reject";
}