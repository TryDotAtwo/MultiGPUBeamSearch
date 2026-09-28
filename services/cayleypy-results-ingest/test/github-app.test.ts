import { decodeJwt, decodeProtectedHeader, exportPKCS8, generateKeyPair } from "jose";
import { beforeAll, beforeEach, describe, expect, test } from "vitest";
import { createAppJwt, getInstallationToken, githubRequest, resetInstallationTokenCacheForTest } from "../src/github-app.js";

const NOW = Date.UTC(2026, 6, 29, 12, 0, 0);
let privateKey = "";
beforeAll(async () => { const pair = await generateKeyPair("RS256", { modulusLength: 2048, extractable: true }); privateKey = await exportPKCS8(pair.privateKey); });
beforeEach(() => resetInstallationTokenCacheForTest());
function env(overrides: Record<string, string | undefined> = {}) { return { GITHUB_APP_ID: "12345", GITHUB_APP_INSTALLATION_ID: "67890", GITHUB_APP_PRIVATE_KEY: privateKey.replace(/\n/g, "\\n"), REPO_OWNER: "TryDotAtwo", REPO_NAME: "cayleypy-beam-results", GITHUB_API_URL: "https://github.example/api/v3/", ...overrides }; }
function tokenResponse(token = "t", expiresAt = new Date(NOW + 120_000).toISOString()) { return new Response(JSON.stringify({ token, expires_at: expiresAt }), { status: 201, headers: { "content-type": "application/json" } }); }

describe("GitHub App JWT and installation token", () => {
  test("uses an RS256 numeric-app JWT, backdated and no longer than ten minutes", async () => {
    const jwt = await createAppJwt(env(), NOW); const header = decodeProtectedHeader(jwt); const claims = decodeJwt(jwt);
    expect(header).toMatchObject({ alg: "RS256", typ: "JWT" }); expect(claims).toMatchObject({ iss: "12345", iat: Math.floor(NOW / 1000) - 30 });
    expect(claims.exp! - claims.iat!).toBeLessThanOrEqual(600); expect(claims.iat).toBeLessThan(Math.floor(NOW / 1000));
  });
  test("POSTs only the installation endpoint, scoped to one repository with least contents permission", async () => {
    let url = ""; let init: RequestInit | undefined;
    await getInstallationToken(env(), NOW, async (input, request) => { url = String(input); init = request; return tokenResponse("x"); });
    expect(url).toBe("https://github.example/api/v3/app/installations/67890/access_tokens"); expect(init?.method).toBe("POST"); expect(new Headers(init?.headers).get("x-github-api-version")).toBe("2026-03-10");
    expect(JSON.parse(String(init?.body))).toEqual({ repositories: ["cayleypy-beam-results"], permissions: { contents: "write", metadata: "read" } });
  });
  test("isolates cache scopes and refreshes before expiry", async () => {
    let calls = 0; const fetcher = async () => tokenResponse(`token-${++calls}`);
    await expect(getInstallationToken(env(), NOW, fetcher)).resolves.toBe("token-1"); await expect(getInstallationToken(env(), NOW + 1, fetcher)).resolves.toBe("token-1");
    await expect(getInstallationToken(env({ REPO_OWNER: "other" }), NOW + 1, fetcher)).resolves.toBe("token-2"); await expect(getInstallationToken(env(), NOW + 60_000, fetcher)).resolves.toBe("token-3"); expect(calls).toBe(3);
  });
  test("rejects malformed/past expiries and non-numeric IDs without network access", async () => {
    await expect(getInstallationToken(env(), NOW, async () => tokenResponse("x", "invalid"))).rejects.toThrow("github_app_auth_failed");
    await expect(getInstallationToken(env(), NOW, async () => tokenResponse("x", new Date(NOW - 1).toISOString()))).rejects.toThrow("github_app_auth_failed");
    let calls = 0; const fetcher = async () => { calls += 1; return tokenResponse(); };
    await expect(getInstallationToken(env({ GITHUB_APP_ID: "abc" }), NOW, fetcher)).rejects.toThrow("github_app_auth_unavailable"); await expect(getInstallationToken(env({ GITHUB_APP_INSTALLATION_ID: "0" }), NOW, fetcher)).rejects.toThrow("github_app_auth_unavailable"); expect(calls).toBe(0);
  });
  test("runs the final callback after signing and before fetch, without calling the network when it stops", async () => {
    let calls = 0; let callbackRan = false;
    await expect(getInstallationToken(env(), NOW, async () => { calls += 1; return tokenResponse(); }, async () => { callbackRan = true; throw new Error("mode-flipped"); })).rejects.toThrow("github_app_auth_failed");
    expect(callbackRan).toBe(true); expect(calls).toBe(0);
  });
  test("redacts sign, network, status, and parse failures", async () => {
    const secret = "PRIVATE-KEY-SENTINEL"; const token = "TOKEN-SENTINEL";
    const attempts = [
      () => createAppJwt(env({ GITHUB_APP_PRIVATE_KEY: secret }), NOW),
      () => getInstallationToken(env(), NOW, async () => { throw new Error(token); }),
      () => getInstallationToken(env(), NOW, async () => new Response(token, { status: 500 })),
      () => getInstallationToken(env(), NOW, async () => new Response("{", { status: 201 })),
    ];
    for (const attempt of attempts) {
      await expect(attempt()).rejects.not.toThrow(secret); await expect(attempt()).rejects.not.toThrow(token);
    }
  });
});

describe("safe GitHub requests", () => {
  test("exposes only status for failures and safely parses successful JSON", async () => {
    await expect(githubRequest(env(), "/repos/a/b", { token: "TOKEN" }, async () => new Response('{"message":"PRIVATE"}', { status: 422 }))).resolves.toEqual({ status: 422, body: undefined });
    await expect(githubRequest(env(), "/repos/a/b", { token: "x" }, async () => new Response('{"sha":"abc"}', { status: 200 }))).resolves.toEqual({ status: 200, body: { sha: "abc" } });
  });
  test("redacts network failure and treats success parse failure safely", async () => {
    await expect(githubRequest(env(), "/x", { token: "TOKEN-SENTINEL" }, async () => { throw new Error("TOKEN-SENTINEL"); })).rejects.toThrow("github_temporary_unavailable");
    await expect(githubRequest(env(), "/x", { token: "x" }, async () => new Response("not-json", { status: 200 }))).resolves.toEqual({ status: 200, body: undefined });
  });
});
