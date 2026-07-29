import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, test } from "vitest";

const runbook = readFileSync(fileURLToPath(new URL("../DEPLOY_STAGING_RUNBOOK.md", import.meta.url)), "utf8");
const script = readFileSync(fileURLToPath(new URL("../scripts/invoke-staging-deployment.ps1", import.meta.url)), "utf8");

describe("staging deployment runbook contract", () => {
  test("is explicit about isolated resources, declarative DO, and activation order", () => {
    expect(runbook).toContain("cayleypy-results-staging"); expect(runbook).toContain("cayleypy-results-raw-staging"); expect(runbook).toContain("cayleypy-validate-staging"); expect(runbook).toContain("cayleypy-validate-dlq-staging"); expect(runbook).toContain("declarative `exports.GitHubWriter`"); expect(runbook).toContain("`store_only`"); expect(runbook).toContain("activate normal"); expect(runbook).toContain("80 unique valid envelopes, 10");
  });
  test("accepts only an exact tool-returned D1 UUID and keeps secrets out of config", () => {
    expect(script).toContain("d1_database_id must be the exact UUID returned"); expect(script).toContain("^[0-9a-fA-F]{8}"); expect(script).toContain("GITHUB_APP_PRIVATE_KEY").toBe(false); expect(runbook).toContain("wrangler secret put GITHUB_APP_PRIVATE_KEY --env staging");
  });
  test("has a fail-closed mode transition and performs migrations before first deploy", () => {
    expect(script).toContain("ValidateSet('preflight', 'store_only', 'activate_normal', 'rollback_store_only')"); expect(script).toContain("$targetMode = if ($Phase -eq 'activate_normal') { 'normal' } else { 'store_only' }"); expect(script).toContain("'d1', 'migrations', 'apply'"); expect(script).toContain("if ($Phase -eq 'store_only')"); expect(script).toContain("Require-CommandResult @('deploy'");
  });
});
