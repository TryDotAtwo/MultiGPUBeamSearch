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
    expect(script).toContain("d1_database_id must be the exact UUID returned"); expect(script).toContain("^[0-9a-fA-F]{8}"); expect(script).not.toContain("GITHUB_APP_PRIVATE_KEY");
    expect(script).toContain("cloudflare_account_id must be the exact 32-hex id");
    expect(script).toContain("'whoami', '--account'");
    expect(runbook).toContain("wrangler.cmd whoami --account $env:CLOUDFLARE_ACCOUNT_ID --json");
    expect(runbook).toContain("if ($LASTEXITCODE -ne 0) { throw 'wrangler whoami failed' }");
    expect(runbook).toContain("if ($LASTEXITCODE -ne 0) { throw 'account-pinned wrangler whoami failed' }");
    expect(runbook).toContain("if ($LASTEXITCODE -ne 0) { throw 'D1 creation failed' }");
    expect(runbook).toContain("if ($LASTEXITCODE -ne 0) { throw 'R2 creation failed' }");
    expect(runbook).toContain("if ($LASTEXITCODE -ne 0) { throw 'validation Queue creation failed' }");
    expect(runbook).toContain("if ($LASTEXITCODE -ne 0) { throw 'DLQ creation failed' }");
    expect(runbook).toContain("Set-Content -LiteralPath .\\staging-resources.private.json -Encoding UTF8 -NoNewline");
    expect(runbook).toContain("Node `22.23.1`");
    expect(runbook).toContain("node --experimental-strip-types .\\test\\recovery-audit.ts");
    expect(runbook).toContain("$env:INGEST_BASE_URL = '<copied from this successful deploy output>'");
    expect(runbook).toContain("--manifest $receiptManifest");
    expect(runbook).toContain("--d1 $d1Snapshot");
    expect(runbook).toContain("--r2 $r2Snapshot");
    expect(runbook).toContain("--github $githubSnapshot");
    expect(runbook).not.toContain("RECOVERY_AUDIT_EXPECTED_MODE");
    expect(runbook).not.toContain("--import tsx");
    expect(runbook).toContain("bootstrap:github-app");
    expect(runbook).toContain("secret bulk");
    expect(runbook).not.toContain("secret put GITHUB_APP_PRIVATE_KEY");
  });
  test("has a fail-closed mode transition and performs migrations before first deploy", () => {
    expect(script).toContain("ValidateSet('preflight', 'store_only', 'activate_normal', 'rollback_store_only')"); expect(script).toContain("$targetMode = if ($Phase -eq 'activate_normal') { 'normal' } else { 'store_only' }"); expect(script).toContain("'d1', 'migrations', 'apply'"); expect(script).toContain("if ($Phase -eq 'store_only')"); expect(script).toContain("Require-CommandResult @('deploy'");
  });
});
