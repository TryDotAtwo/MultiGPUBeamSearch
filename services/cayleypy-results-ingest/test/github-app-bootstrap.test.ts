import { generateKeyPairSync } from "node:crypto";
import { decodeJwt, decodeProtectedHeader } from "jose";
import { describe, expect, test, vi } from "vitest";
import {
  CONTRACT,
  buildExactCloudflareEnvironment,
  buildGitHubAppManifest,
  buildSecretBulkInvocation,
  convertPkcs1ToPkcs8,
  createAppJwt,
  createManifestState,
  parseArguments,
  runBootstrap,
  startManifestFlow,
  validateCallbackRequest,
  validateCloudflareAccountOverride,
  validateConversion,
  validateEmptySecretList,
  validateGeneratedStoreOnlyConfig,
  validateSecretList,
  verifyExactInstallation,
  waitForInstallation,
} from "../scripts/github-app-bootstrap.mjs";

const CONFIG_PATH = "D:\\safe root\\services\\cayleypy-results-ingest\\.staging-deploy-private\\wrangler.generated.json";
const SERVICE_ROOT = "D:\\safe root\\services\\cayleypy-results-ingest";
const D1_ID = "11111111-2222-4333-8444-555555555555";
const ACCOUNT_ID = "0123456789abcdef0123456789abcdef";

function generatedConfig(mode = "store_only") {
  return {
    name: "cayleypy-results-ingest",
    account_id: ACCOUNT_ID,
    env: { staging: {
      vars: { INGEST_MODE: mode, REPO_OWNER: "TryDotAtwo", REPO_NAME: "cayleypy-beam-results", STAGING_BRANCH: "ingest/staging" },
      d1_databases: [{ binding: "RESULTS_DB", database_name: "cayleypy-results-staging", database_id: D1_ID, migrations_dir: `${SERVICE_ROOT}\\migrations` }],
      r2_buckets: [{ binding: "RAW_RESULTS", bucket_name: "cayleypy-results-raw-staging" }],
      queues: {
        producers: [
          { binding: "VALIDATE_QUEUE", queue: "cayleypy-validate-staging" },
          { binding: "VALIDATE_DLQ", queue: "cayleypy-validate-dlq-staging" },
        ],
        consumers: [{ queue: "cayleypy-validate-staging", max_batch_size: 10, max_retries: 8, dead_letter_queue: "cayleypy-validate-dlq-staging" }],
      },
    } },
  };
}

function exactInstallation() {
  return {
    id: 789, app_id: 456,
    account: { login: "TryDotAtwo", type: "User" },
    repository_selection: "selected",
    permissions: { contents: "write", metadata: "read" },
    events: [], suspended_at: null,
  };
}

describe("GitHub App manifest bootstrap contracts", () => {
  test("keeps the help parse result endpoint-free", () => {
    expect(parseArguments(["--help"])).toEqual({
      dryRun: false,
      confirmCreatePrivateApp: false,
      stagingEndpoint: undefined,
      help: true,
    });
  });

  test("uses 32 random bytes and the personal-account least-privilege manifest", () => {
    const randomBytes = vi.fn((length: number) => {
      expect(length).toBe(32);
      return new Uint8Array(Array.from({ length }, (_, index) => index + 1));
    });
    expect(createManifestState(randomBytes)).toMatch(/^[A-Za-z0-9_-]{43}$/);
    expect(buildGitHubAppManifest("http://127.0.0.1:43123/github-app-manifest/callback")).toEqual({
      name: "cayleypy-beam-results-ingest",
      url: "https://github.com/TryDotAtwo/cayleypy-beam-results",
      redirect_url: "http://127.0.0.1:43123/github-app-manifest/callback",
      hook_attributes: { url: "https://github.com/TryDotAtwo/cayleypy-beam-results", active: false },
      public: false,
      default_permissions: { contents: "write", metadata: "read" },
      default_events: [], request_oauth_on_install: false, setup_on_update: false,
    });
  });

  test("accepts only one exact callback and redacts rejected query values", () => {
    const expectedState = "A".repeat(43);
    expect(validateCallbackRequest({ method: "GET", requestUrl: `/github-app-manifest/callback?code=${"b".repeat(40)}&state=${expectedState}`, expectedState })).toBe("b".repeat(40));
    const code = "SECRET_CALLBACK_CODE_1234567890";
    const attempts = [
      `/github-app-manifest/callback?code=${code}&state=wrong`,
      `/github-app-manifest/callback?code=${code}&code=duplicate&state=${expectedState}`,
      `/github-app-manifest/callback?code=${code}&state=${expectedState}&state=duplicate`,
      `/github-app-manifest/callback?code=${code}&state=${expectedState}&extra=1`,
      `/wrong?code=${code}&state=${expectedState}`,
    ];
    for (const requestUrl of attempts) {
      expect(() => validateCallbackRequest({ method: "GET", requestUrl, expectedState })).toThrow("github_app_callback_rejected");
      try { validateCallbackRequest({ method: "GET", requestUrl, expectedState }); }
      catch (error) { expect(String(error)).not.toContain(code); expect(String(error)).not.toContain("wrong"); }
    }
    expect(() => validateCallbackRequest({ method: "POST", requestUrl: `/github-app-manifest/callback?code=${code}&state=${expectedState}`, expectedState })).toThrow("github_app_callback_rejected");
  });

  test("converts GitHub PKCS#1 to PKCS#8 in memory and redacts failures", () => {
    const { privateKey } = generateKeyPairSync("rsa", { modulusLength: 2048 });
    const pkcs1 = privateKey.export({ type: "pkcs1", format: "pem" }).toString();
    const pkcs8 = convertPkcs1ToPkcs8(pkcs1);
    const jwt = createAppJwt({ appId: 456, privateKeyPkcs8: pkcs8, nowMs: 1_800_000_000_000 });
    expect(pkcs1).toContain("BEGIN RSA PRIVATE KEY");
    expect(pkcs8).toContain("BEGIN PRIVATE KEY");
    expect(pkcs8).not.toContain("BEGIN RSA PRIVATE KEY");
    expect(decodeProtectedHeader(jwt)).toEqual({ alg: "RS256", typ: "JWT" });
    expect(decodeJwt(jwt)).toMatchObject({ iss: "456", iat: 1_799_999_970, exp: 1_800_000_540 });
    expect(jwt).not.toContain("BEGIN PRIVATE KEY");
    const sentinel = "PRIVATE_KEY_SENTINEL";
    expect(() => convertPkcs1ToPkcs8(sentinel)).toThrow("github_app_private_key_invalid");
    try { convertPkcs1ToPkcs8(sentinel); } catch (error) { expect(String(error)).not.toContain(sentinel); }
  });

  test("validates only the fixed private generated store-only sink", () => {
    expect(validateGeneratedStoreOnlyConfig({ config: generatedConfig(), configPath: CONFIG_PATH, serviceRoot: SERVICE_ROOT })).toEqual({
      accountId: ACCOUNT_ID,
      d1DatabaseId: D1_ID,
    });
    for (const config of [
      generatedConfig("normal"),
      { ...generatedConfig(), name: "other-worker" },
      { ...generatedConfig(), account_id: "not-an-account-id" },
      { ...generatedConfig(), account_id: undefined },
      { ...generatedConfig(), legacy_env: true },
      { ...generatedConfig(), env: { staging: { ...generatedConfig().env.staging, name: "other-worker" } } },
      { ...generatedConfig(), env: { staging: { ...generatedConfig().env.staging, account_id: ACCOUNT_ID } } },
      { ...generatedConfig(), env: { staging: { ...generatedConfig().env.staging, legacy_env: true } } },
      { ...generatedConfig(), env: { staging: { ...generatedConfig().env.staging, vars: { ...generatedConfig().env.staging.vars, REPO_NAME: "other" } } } },
    ]) {
      expect(() => validateGeneratedStoreOnlyConfig({ config, configPath: CONFIG_PATH, serviceRoot: SERVICE_ROOT })).toThrow("staging_secret_sink_invalid");
    }
    expect(() => validateGeneratedStoreOnlyConfig({ config: generatedConfig(), configPath: "D:\\other\\wrangler.generated.json", serviceRoot: SERVICE_ROOT })).toThrow("staging_secret_sink_invalid");
  });

  test("requires one private App installation on exactly the personal repo", () => {
    expect(validateConversion({
      id: 456, name: "cayleypy-beam-results-ingest",
      owner: { login: "TryDotAtwo", type: "User" },
      html_url: "https://github.com/apps/cayleypy-beam-results-ingest",
      public: false, permissions: { contents: "write", metadata: "read" }, events: [],
      pem: "-----BEGIN RSA PRIVATE KEY-----\nTEST\n-----END RSA PRIVATE KEY-----",
      client_secret: "discard-me", webhook_secret: "discard-me-too",
    })).toMatchObject({ appId: 456, installUrl: "https://github.com/apps/cayleypy-beam-results-ingest/installations/new" });
    expect(() => validateConversion({
      id: 456, name: "cayleypy-beam-results-ingest", owner: { login: "TryDotAtwo", type: "User" },
      html_url: "https://github.com/apps/lookalike-app", public: false,
      permissions: { contents: "write", metadata: "read" }, events: [],
      pem: "-----BEGIN RSA PRIVATE KEY-----\nTEST\n-----END RSA PRIVATE KEY-----",
    })).toThrow("github_app_conversion_invalid");
    expect(verifyExactInstallation({
      appId: 456,
      installation: exactInstallation(),
      repositoryResponse: { total_count: 1, repositories: [{ id: 1_281_329_788, full_name: "TryDotAtwo/cayleypy-beam-results" }] },
    })).toEqual({ installationId: 789 });
    const wrong = { ...exactInstallation(), permissions: { contents: "write", metadata: "read", issues: "read" } };
    expect(() => verifyExactInstallation({ appId: 456, installation: wrong, repositoryResponse: { total_count: 1, repositories: [{ id: 1_281_329_788, full_name: "TryDotAtwo/cayleypy-beam-results" }] } })).toThrow("github_app_installation_invalid");
    expect(() => verifyExactInstallation({ appId: 456, installation: exactInstallation(), repositoryResponse: { total_count: 2, repositories: [{ id: 1_281_329_788, full_name: "TryDotAtwo/cayleypy-beam-results" }, { id: 999, full_name: "TryDotAtwo/other" }] } })).toThrow("github_app_installation_invalid");
  });

  test("verifies the full selected repository set with App auth and revokes its read-only token", async () => {
    const { privateKey } = generateKeyPairSync("rsa", { modulusLength: 2048 });
    const pkcs8 = convertPkcs1ToPkcs8(privateKey.export({ type: "pkcs1", format: "pem" }).toString());
    const installationToken = `ghs_${"t".repeat(40)}`;
    const fetchMock = vi.fn(async (input: unknown, init?: RequestInit) => {
      const call = fetchMock.mock.calls.length;
      const url = String(input);
      const authorization = (init?.headers as Record<string, string>)?.authorization;
      expect(init?.redirect).toBe("error");
      if (call <= 3) expect(authorization).toMatch(/^Bearer eyJ/);
      if (call >= 4) expect(authorization).toBe(`Bearer ${installationToken}`);
      if (call === 1) {
        expect(url).toBe("https://api.github.com/app/installations?per_page=100");
        return new Response(JSON.stringify([exactInstallation()]), { status: 200 });
      }
      if (call === 2) {
        expect(url).toBe("https://api.github.com/repos/TryDotAtwo/cayleypy-beam-results/installation");
        return new Response(JSON.stringify(exactInstallation()), { status: 200 });
      }
      if (call === 3) {
        expect(url).toBe("https://api.github.com/app/installations/789/access_tokens");
        expect(init?.method).toBe("POST");
        expect(JSON.parse(String(init?.body))).toEqual({ permissions: { contents: "read", metadata: "read" } });
        return new Response(JSON.stringify({
          token: installationToken,
          expires_at: new Date(Date.now() + 3_600_000).toISOString(),
          repository_selection: "selected",
          permissions: { contents: "read", metadata: "read" },
        }), { status: 201 });
      }
      if (call === 4) {
        expect(url).toBe("https://api.github.com/installation/repositories?per_page=100");
        return new Response(JSON.stringify({ total_count: 1, repositories: [{ id: 1_281_329_788, full_name: "TryDotAtwo/cayleypy-beam-results" }] }), { status: 200 });
      }
      expect(call).toBe(5);
      expect(url).toBe("https://api.github.com/installation/token");
      expect(init?.method).toBe("DELETE");
      return new Response(null, { status: 204 });
    });
    vi.stubGlobal("fetch", fetchMock);
    try {
      await expect(waitForInstallation(456, pkcs8)).resolves.toEqual({ installationId: 789 });
      expect(fetchMock).toHaveBeenCalledTimes(5);
    } finally { vi.unstubAllGlobals(); }
  });

  test("revokes the temporary token before failing a repository-selection check", async () => {
    const { privateKey } = generateKeyPairSync("rsa", { modulusLength: 2048 });
    const pkcs8 = convertPkcs1ToPkcs8(privateKey.export({ type: "pkcs1", format: "pem" }).toString());
    const installationToken = `ghs_${"r".repeat(40)}`;
    const fetchMock = vi.fn(async (_input: unknown, init?: RequestInit) => {
      const call = fetchMock.mock.calls.length;
      if (call === 1) return new Response(JSON.stringify([exactInstallation()]), { status: 200 });
      if (call === 2) return new Response(JSON.stringify(exactInstallation()), { status: 200 });
      if (call === 3) return new Response(JSON.stringify({
        token: installationToken,
        expires_at: new Date(Date.now() + 3_600_000).toISOString(),
        repository_selection: "selected",
        permissions: { contents: "read", metadata: "read" },
      }), { status: 201 });
      if (call === 4) return new Response(JSON.stringify({ total_count: 2, repositories: [
        { id: 1_281_329_788, full_name: "TryDotAtwo/cayleypy-beam-results" },
        { id: 9, full_name: "TryDotAtwo/other" },
      ] }), { status: 200 });
      expect((init?.headers as Record<string, string>)?.authorization).toBe(`Bearer ${installationToken}`);
      expect(init?.method).toBe("DELETE");
      return new Response(null, { status: 204 });
    });
    vi.stubGlobal("fetch", fetchMock);
    try {
      await expect(waitForInstallation(456, pkcs8)).rejects.toThrow("github_app_installation_invalid");
      expect(fetchMock).toHaveBeenCalledTimes(5);
    } finally { vi.unstubAllGlobals(); }
  });
  test("fails closed on paginated App installation enumeration", async () => {
    const { privateKey } = generateKeyPairSync("rsa", { modulusLength: 2048 });
    const pkcs8 = convertPkcs1ToPkcs8(privateKey.export({ type: "pkcs1", format: "pem" }).toString());
    const fetchMock = vi.fn(async () => new Response(JSON.stringify([exactInstallation()]), {
      status: 200,
      headers: { link: '<https://api.github.com/app/installations?page=2>; rel="next"' },
    }));
    vi.stubGlobal("fetch", fetchMock);
    try {
      await expect(waitForInstallation(456, pkcs8)).rejects.toThrow("github_app_installation_invalid");
      expect(fetchMock).toHaveBeenCalledTimes(1);
    } finally { vi.unstubAllGlobals(); }
  });

  test("rejects repository-list pagination and still revokes the temporary token", async () => {
    const { privateKey } = generateKeyPairSync("rsa", { modulusLength: 2048 });
    const pkcs8 = convertPkcs1ToPkcs8(privateKey.export({ type: "pkcs1", format: "pem" }).toString());
    const installationToken = `ghs_${"p".repeat(40)}`;
    const fetchMock = vi.fn(async (_input: unknown, init?: RequestInit) => {
      const call = fetchMock.mock.calls.length;
      if (call === 1) return new Response(JSON.stringify([exactInstallation()]), { status: 200 });
      if (call === 2) return new Response(JSON.stringify(exactInstallation()), { status: 200 });
      if (call === 3) return new Response(JSON.stringify({
        token: installationToken,
        expires_at: new Date(Date.now() + 3_600_000).toISOString(),
        repository_selection: "selected",
        permissions: { contents: "read", metadata: "read" },
      }), { status: 201 });
      if (call === 4) return new Response(JSON.stringify({ total_count: 1, repositories: [{ id: 1_281_329_788, full_name: "TryDotAtwo/cayleypy-beam-results" }] }), {
        status: 200,
        headers: { link: '<https://api.github.com/installation/repositories?page=2>; rel="next"' },
      });
      expect(call).toBe(5);
      expect(init?.method).toBe("DELETE");
      return new Response(null, { status: 204 });
    });
    vi.stubGlobal("fetch", fetchMock);
    try {
      await expect(waitForInstallation(456, pkcs8)).rejects.toThrow("github_app_installation_invalid");
      expect(fetchMock).toHaveBeenCalledTimes(5);
    } finally { vi.unstubAllGlobals(); }
  });

  test("revokes temporary tokens with any selection, permission, or expiry drift", async () => {
    const { privateKey } = generateKeyPairSync("rsa", { modulusLength: 2048 });
    const pkcs8 = convertPkcs1ToPkcs8(privateKey.export({ type: "pkcs1", format: "pem" }).toString());
    const validExpiry = new Date(Date.now() + 3_600_000).toISOString();
    for (const tokenClaims of [
      { permissions: { contents: "write", metadata: "read" }, expires_at: validExpiry },
      { permissions: { contents: "read" }, expires_at: validExpiry },
      { permissions: { contents: "read", metadata: "read", issues: "read" }, expires_at: validExpiry },
      { permissions: { contents: "read", metadata: "read" }, repository_selection: "all", expires_at: validExpiry },
      { permissions: { contents: "read", metadata: "read" }, expires_at: new Date(Date.now() - 1_000).toISOString() },
      { permissions: { contents: "read", metadata: "read" }, expires_at: "not-a-date" },
      { permissions: { contents: "read", metadata: "read" }, expires_at: new Date(Date.now() + 7_200_000).toISOString() },
    ]) {
      const installationToken = `ghs_${"s".repeat(40)}`;
      const fetchMock = vi.fn(async (_input: unknown, init?: RequestInit) => {
        const call = fetchMock.mock.calls.length;
        if (call === 1) return new Response(JSON.stringify([exactInstallation()]), { status: 200 });
        if (call === 2) return new Response(JSON.stringify(exactInstallation()), { status: 200 });
        if (call === 3) return new Response(JSON.stringify({
          token: installationToken,
          repository_selection: "selected",
          ...tokenClaims,
        }), { status: 201 });
        expect(call).toBe(4);
        expect(init?.method).toBe("DELETE");
        return new Response(null, { status: 204 });
      });
      vi.stubGlobal("fetch", fetchMock);
      try {
        await expect(waitForInstallation(456, pkcs8)).rejects.toThrow("github_app_installation_invalid");
        expect(fetchMock).toHaveBeenCalledTimes(4);
      } finally { vi.unstubAllGlobals(); }
    }
  });

  test("treats temporary-token revocation failure as a bootstrap failure", async () => {
    const { privateKey } = generateKeyPairSync("rsa", { modulusLength: 2048 });
    const pkcs8 = convertPkcs1ToPkcs8(privateKey.export({ type: "pkcs1", format: "pem" }).toString());
    const installationToken = `ghs_${"v".repeat(40)}`;
    const fetchMock = vi.fn(async (_input: unknown, init?: RequestInit) => {
      const call = fetchMock.mock.calls.length;
      if (call === 1) return new Response(JSON.stringify([exactInstallation()]), { status: 200 });
      if (call === 2) return new Response(JSON.stringify(exactInstallation()), { status: 200 });
      if (call === 3) return new Response(JSON.stringify({
        token: installationToken,
        expires_at: new Date(Date.now() + 3_600_000).toISOString(),
        repository_selection: "selected",
        permissions: { contents: "read", metadata: "read" },
      }), { status: 201 });
      if (call === 4) return new Response(JSON.stringify({ total_count: 1, repositories: [{ id: 1_281_329_788, full_name: "TryDotAtwo/cayleypy-beam-results" }] }), { status: 200 });
      expect(call).toBe(5);
      expect(init?.method).toBe("DELETE");
      return new Response(JSON.stringify({ message: "unavailable" }), { status: 503 });
    });
    vi.stubGlobal("fetch", fetchMock);
    try {
      await expect(waitForInstallation(456, pkcs8)).rejects.toThrow("github_app_installation_lookup_failed");
      expect(fetchMock).toHaveBeenCalledTimes(5);
    } finally { vi.unstubAllGlobals(); }
  });

  test("passes exactly three secrets on stdin to pinned Wrangler bulk", () => {
    const invocation = buildSecretBulkInvocation({
      wranglerEntrypoint: "D:\\safe root\\node_modules\\wrangler\\bin\\wrangler.js",
      generatedConfigPath: CONFIG_PATH, appId: 456, installationId: 789,
      privateKeyPkcs8: "-----BEGIN PRIVATE KEY-----\nPRIVATE\n-----END PRIVATE KEY-----",
    });
    expect(invocation.command).toMatch(/node(?:\.exe)?$/i);
    expect(invocation.args).toEqual(["D:\\safe root\\node_modules\\wrangler\\bin\\wrangler.js", "secret", "bulk", "--config", CONFIG_PATH, "--env", "staging"]);
    expect(JSON.parse(invocation.stdin)).toEqual({
      GITHUB_APP_ID: "456", GITHUB_APP_INSTALLATION_ID: "789",
      GITHUB_APP_PRIVATE_KEY: "-----BEGIN PRIVATE KEY-----\nPRIVATE\n-----END PRIVATE KEY-----",
    });
    expect(invocation.args.join(" ")).not.toContain("PRIVATE");
  });

  test("accepts exactly the three expected staging secret names", () => {
    const exact = [
      { name: "GITHUB_APP_PRIVATE_KEY", type: "secret_text" },
      { name: "GITHUB_APP_ID", type: "secret_text" },
      { name: "GITHUB_APP_INSTALLATION_ID", type: "secret_text" },
    ];
    expect(validateSecretList(exact)).toEqual(["GITHUB_APP_ID", "GITHUB_APP_INSTALLATION_ID", "GITHUB_APP_PRIVATE_KEY"]);
    expect(() => validateSecretList([...exact, { name: "UNEXPECTED_SECRET" }])).toThrow("secret_bulk_verification_failed");
    expect(() => validateSecretList(exact.slice(0, 2))).toThrow("secret_bulk_verification_failed");
    expect(() => validateSecretList([{ name: "GITHUB_APP_ID" }, { name: "GITHUB_APP_ID" }, { name: "GITHUB_APP_PRIVATE_KEY" }])).toThrow("secret_bulk_verification_failed");
    expect(() => validateSecretList(exact.map((entry) => entry.name === "GITHUB_APP_ID" ? { ...entry, type: "plain_text" } : entry))).toThrow("secret_bulk_verification_failed");
  });

  test("requires an empty staging secret set before any App or bulk mutation", () => {
    expect(validateEmptySecretList([])).toEqual([]);
    expect(() => validateEmptySecretList([{ name: "OLD_SECRET", type: "secret_text" }])).toThrow("staging_preexisting_secrets_forbidden");
    expect(() => validateEmptySecretList({})).toThrow("staging_preexisting_secrets_forbidden");
  });

  test("rejects any Cloudflare account environment override that conflicts with the generated config", () => {
    expect(validateCloudflareAccountOverride(undefined, ACCOUNT_ID)).toBe(ACCOUNT_ID);
    expect(validateCloudflareAccountOverride(ACCOUNT_ID.toUpperCase(), ACCOUNT_ID)).toBe(ACCOUNT_ID);
    expect(() => validateCloudflareAccountOverride("f".repeat(32), ACCOUNT_ID)).toThrow("cloudflare_account_mismatch");
    expect(() => validateCloudflareAccountOverride(undefined, "not-an-account-id")).toThrow("cloudflare_account_mismatch");
  });

  test("canonicalizes every case-insensitive Windows Cloudflare account environment alias", () => {
    const sanitized = buildExactCloudflareEnvironment({
      Path: "safe-path",
      cloudflare_account_id: ACCOUNT_ID.toUpperCase(),
      CloudFlare_Account_Id: ACCOUNT_ID,
    }, ACCOUNT_ID);
    expect(sanitized).toEqual({ Path: "safe-path", CLOUDFLARE_ACCOUNT_ID: ACCOUNT_ID });
    expect(() => buildExactCloudflareEnvironment({ cloudflare_account_id: "f".repeat(32) }, ACCOUNT_ID)).toThrow("cloudflare_account_mismatch");
  });
  test("dry-run verifies prerequisites but performs no mutation", async () => {
    const calls: string[] = [];
    const deps = {
      verifyPrerequisites: vi.fn(async () => { calls.push("preflight"); }),
      startManifestFlow: vi.fn(async () => { calls.push("listener"); throw new Error("must not run"); }),
      exchangeManifest: vi.fn(async () => { calls.push("post"); throw new Error("must not run"); }),
      openBrowser: vi.fn(async () => { calls.push("browser"); }),
      waitForInstallation: vi.fn(async () => { calls.push("installation"); throw new Error("must not run"); }),
      uploadSecrets: vi.fn(async () => { calls.push("secret-bulk"); }),
      report: vi.fn((message: string) => expect(message).not.toMatch(/PRIVATE|BEGIN|state|code|token/i)),
    };
    await expect(runBootstrap({ dryRun: true, stagingEndpoint: "https://staging.example.test" }, deps)).resolves.toEqual({ status: "dry_run_ok" });
    expect(calls).toEqual(["preflight"]);
    expect(deps.startManifestFlow).not.toHaveBeenCalled();
    expect(deps.exchangeManifest).not.toHaveBeenCalled();
    expect(deps.openBrowser).not.toHaveBeenCalled();
    expect(deps.uploadSecrets).not.toHaveBeenCalled();
  });

  test("a failed initial precheck prevents every mutation path", async () => {
    const deps = {
      verifyPrerequisites: vi.fn(async () => { throw new Error("staging_preexisting_secrets_forbidden"); }),
      startManifestFlow: vi.fn(async () => { throw new Error("must_not_run"); }),
      exchangeManifest: vi.fn(async (_code: string) => ({})),
      openBrowser: vi.fn(async (_url: string) => {}),
      waitForInstallation: vi.fn(async (_appId: number, _privateKeyPkcs8: string) => ({ installationId: 789 })),
      uploadSecrets: vi.fn(async (_input: { appId: number; installationId: number; privateKeyPkcs8: string }) => {}),
      report: vi.fn((_message: string) => {}),
    };
    await expect(runBootstrap({ dryRun: false, confirmCreatePrivateApp: true, stagingEndpoint: "https://staging.example.test" }, deps)).rejects.toThrow("staging_preexisting_secrets_forbidden");
    expect(deps.startManifestFlow).not.toHaveBeenCalled();
    expect(deps.exchangeManifest).not.toHaveBeenCalled();
    expect(deps.openBrowser).not.toHaveBeenCalled();
    expect(deps.uploadSecrets).not.toHaveBeenCalled();
  });

  test("rechecks the store-only sink before one bulk upload and clears conversion secrets", async () => {
    const { privateKey } = generateKeyPairSync("rsa", { modulusLength: 2048 });
    const pkcs1 = privateKey.export({ type: "pkcs1", format: "pem" }).toString();
    const conversion = {
      id: 456, name: "cayleypy-beam-results-ingest",
      owner: { login: "TryDotAtwo", type: "User" },
      html_url: "https://github.com/apps/cayleypy-beam-results-ingest",
      public: false, permissions: { contents: "write", metadata: "read" }, events: [],
      pem: pkcs1, client_secret: "client-secret", webhook_secret: "webhook-secret",
    };
    const calls: string[] = [];
    const close = vi.fn(() => calls.push("close"));
    const deps = {
      verifyPrerequisites: vi.fn(async ({ finalCheck }: { finalCheck: boolean }) => { calls.push(finalCheck ? "preflight-final" : "preflight-initial"); }),
      startManifestFlow: vi.fn(async () => ({ startUrl: "http://127.0.0.1:43123/github-app-manifest/start", code: Promise.resolve("c".repeat(40)), close })),
      exchangeManifest: vi.fn(async () => { calls.push("exchange"); return conversion; }),
      openBrowser: vi.fn(async (url: string) => { calls.push(url.includes("installations/new") ? "browser-install" : "browser-manifest"); }),
      waitForInstallation: vi.fn(async (appId: number, privateKeyPkcs8: string) => {
        calls.push("verify-installation");
        expect(appId).toBe(456);
        expect(privateKeyPkcs8).toMatch(/^-----BEGIN PRIVATE KEY-----/);
        return { installationId: 789 };
      }),
      uploadSecrets: vi.fn(async ({ appId, installationId, privateKeyPkcs8 }: { appId: number; installationId: number; privateKeyPkcs8: string }) => {
        calls.push("secret-bulk");
        expect({ appId, installationId }).toEqual({ appId: 456, installationId: 789 });
        expect(privateKeyPkcs8).toMatch(/^-----BEGIN PRIVATE KEY-----/);
      }),
      report: vi.fn(),
    };
    await expect(runBootstrap({ dryRun: false, confirmCreatePrivateApp: true, stagingEndpoint: "https://staging.example.test" }, deps)).resolves.toEqual({ status: "ok", appId: 456, installationId: 789 });
    expect(calls).toEqual(["preflight-initial", "browser-manifest", "exchange", "browser-install", "verify-installation", "preflight-final", "secret-bulk", "close"]);
    expect(deps.uploadSecrets).toHaveBeenCalledTimes(1);
    expect(conversion.pem).toBe("");
    expect(conversion.client_secret).toBe("");
    expect(conversion.webhook_secret).toBe("");
  });
  test("the real one-shot listener binds only IPv4 loopback and consumes one exact callback", async () => {
    const flow = await startManifestFlow();
    try {
      const start = new URL(flow.startUrl);
      expect(start.hostname).toBe("127.0.0.1");
      expect(start.pathname).toBe("/github-app-manifest/start");
      const page = await fetch(flow.startUrl);
      const html = await page.text();
      const state = html.match(/settings\/apps\/new\?state=([A-Za-z0-9_-]{43})/)?.[1];
      expect(state).toBeDefined();
      const rejected = new URL("/github-app-manifest/callback", start);
      const wrongState = `${state![0] === "A" ? "B" : "A"}${state!.slice(1)}`;
      rejected.search = new URLSearchParams({ code: "x".repeat(40), state: wrongState }).toString();
      expect((await fetch(rejected)).status).toBe(400);

      const code = "c".repeat(40);
      const callback = new URL("/github-app-manifest/callback", start);
      callback.search = new URLSearchParams({ code, state: state! }).toString();
      expect((await fetch(callback)).status).toBe(200);
      await expect(flow.code).resolves.toBe(code);
    } finally { flow.close(); }
  });

  test("pins owner, repository, version, and IPv4 loopback", () => {
    expect(CONTRACT).toMatchObject({ loopbackHost: "127.0.0.1", wranglerVersion: "4.115.0", owner: "TryDotAtwo", repository: "cayleypy-beam-results", repositoryId: 1_281_329_788, environment: "staging" });
  });
});
