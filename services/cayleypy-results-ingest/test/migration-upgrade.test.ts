import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, test } from "vitest";

const migration = readFileSync(fileURLToPath(new URL("../migrations/0003_remove_legacy_status_ip_limits.sql", import.meta.url)), "utf8");

describe("status limiter upgrade migration", () => {
  test("deletes only legacy status-IP rows", () => {
    expect(migration.trim()).toBe("DELETE FROM ingest_rate_limits WHERE scope LIKE 'status-ip:%';");
    expect(migration).not.toContain("DELETE FROM ingest_rate_limits;");
  });
});