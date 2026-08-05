import { spawn } from "node:child_process";
import { randomBytes } from "node:crypto";
import { lstat, readFile } from "node:fs/promises";
import { createServer } from "node:http";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { EnvHttpProxyAgent, setGlobalDispatcher } from "undici";

import {
  CONTRACT,
  buildExactCloudflareEnvironment,
  buildGitHubAppManifest,
  buildSecretBulkInvocation,
  createAppJwt,
  createManifestState,
  runBootstrap,
  validateCallbackRequest,
  validateEmptySecretList,
  validateGeneratedStoreOnlyConfig,
  validateSecretList,
  validateStagingEndpoint,
  verifyExactInstallation,
} from "./github-app-bootstrap-core.mjs";

export * from "./github-app-bootstrap-core.mjs";

if (process.env.HTTPS_PROXY || process.env.HTTP_PROXY || process.env.ALL_PROXY) {
  setGlobalDispatcher(new EnvHttpProxyAgent());
}

const SERVICE_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const GENERATED_CONFIG = path.join(SERVICE_ROOT, ".staging-deploy-private", "wrangler.generated.json");
const WRANGLER_PACKAGE = path.join(SERVICE_ROOT, "node_modules", "wrangler", "package.json");
const WRANGLER_ENTRYPOINT = path.join(SERVICE_ROOT, "node_modules", "wrangler", "bin", "wrangler.js");
const MAX_OUTPUT_BYTES = 1024 * 1024;
const CALLBACK_TIMEOUT_MS = 15 * 60_000;
const INSTALL_TIMEOUT_MS = 10 * 60_000;

function safeFail(code) { throw new Error(code); }
function safeCode(error) {
  return error instanceof Error && /^[a-z0-9_]+$/.test(error.message) ? error.message : "github_app_bootstrap_failed";
}

async function boundedText(response, limit, code) {
  const declared = Number(response.headers.get("content-length"));
  if (Number.isFinite(declared) && declared > limit) safeFail(code);
  const text = await response.text();
  if (Buffer.byteLength(text, "utf8") > limit) safeFail(code);
  return text;
}

export function runCommand(command, args, { stdin, capture = true, timeoutMs = 60_000, environment } = {}) {
  return new Promise((resolve, reject) => {
    let settled = false;
    let stdout = "";
    let outputBytes = 0;
    let child;
    let timer;
    const done = (error, result) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      if (error) reject(error); else resolve(result);
    };
    try {
      child = spawn(command, args, { shell: false, windowsHide: true, stdio: ["pipe", "pipe", "pipe"], env: environment });
    } catch { safeFail("bootstrap_command_failed"); }
    const onData = (chunk, isStdout) => {
      outputBytes += chunk.byteLength;
      if (outputBytes > MAX_OUTPUT_BYTES) {
        child.kill();
        done(new Error("bootstrap_command_failed"));
        return;
      }
      if (capture && isStdout) stdout += chunk.toString("utf8");
    };
    child.stdout.on("data", (chunk) => onData(chunk, true));
    child.stderr.on("data", (chunk) => onData(chunk, false));
    child.once("error", () => done(new Error("bootstrap_command_failed")));
    child.once("close", (code) => code === 0 ? done(undefined, { stdout }) : done(new Error("bootstrap_command_failed")));
    timer = setTimeout(() => { child.kill(); done(new Error("bootstrap_command_failed")); }, timeoutMs);
    if (stdin === undefined) child.stdin.end(); else child.stdin.end(stdin, "utf8");
  });
}

async function readExactJson(file, code) {
  try {
    const stat = await lstat(file);
    if (!stat.isFile() || stat.isSymbolicLink() || stat.size <= 0 || stat.size > MAX_OUTPUT_BYTES) safeFail(code);
    return JSON.parse(await readFile(file, "utf8"));
  } catch { safeFail(code); }
}

async function verifyStoreOnlyHealth(stagingEndpoint) {
  const origin = validateStagingEndpoint(stagingEndpoint);
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 15_000);
  try {
    const response = await fetch(`${origin}/healthz`, {
      method: "GET",
      headers: { accept: "application/json" },
      redirect: "error",
      signal: controller.signal,
    });
    if (response.status !== 200) safeFail("staging_store_only_health_failed");
    const body = JSON.parse(await boundedText(response, 32 * 1024, "staging_store_only_health_failed"));
    if (body?.status !== "ok" || body?.ingest_mode !== "store_only") safeFail("staging_store_only_health_failed");
  } catch { safeFail("staging_store_only_health_failed"); }
  finally { clearTimeout(timer); }
}

function parseJsonOutput(output, code) {
  try { return JSON.parse(output); } catch { safeFail(code); }
}

async function readAndValidateGeneratedConfig() {
  const generated = await readExactJson(GENERATED_CONFIG, "staging_secret_sink_invalid");
  return validateGeneratedStoreOnlyConfig({ config: generated, configPath: GENERATED_CONFIG, serviceRoot: SERVICE_ROOT });
}

function exactWranglerEnvironment(accountId) {
  return buildExactCloudflareEnvironment(process.env, accountId);
}

function runWrangler(args, accountId, options = {}) {
  return runCommand(process.execPath, [WRANGLER_ENTRYPOINT, ...args], {
    ...options,
    environment: exactWranglerEnvironment(accountId),
  });
}

async function readStagingSecretList(code, accountId) {
  const listed = await runWrangler([
    "secret", "list", "--format", "json",
    "--config", GENERATED_CONFIG, "--env", CONTRACT.environment,
  ], accountId);
  return parseJsonOutput(listed.stdout, code);
}
async function verifyPrerequisites({ stagingEndpoint }) {
  const { accountId } = await readAndValidateGeneratedConfig();
  const wranglerPackage = await readExactJson(WRANGLER_PACKAGE, "pinned_wrangler_unavailable");
  if (wranglerPackage?.version !== CONTRACT.wranglerVersion) safeFail("pinned_wrangler_unavailable");
  const wranglerEntrypoint = await lstat(WRANGLER_ENTRYPOINT).catch(() => undefined);
  if (!wranglerEntrypoint?.isFile()) safeFail("pinned_wrangler_unavailable");

  const version = await runWrangler(["--version"], accountId);
  const versionMatches = String(version.stdout).match(/\b\d+\.\d+\.\d+\b/g) ?? [];
  if (!versionMatches.includes(CONTRACT.wranglerVersion)) safeFail("pinned_wrangler_unavailable");
  await runWrangler(["check", "--config", GENERATED_CONFIG, "--env", CONTRACT.environment], accountId);
  await runWrangler(["whoami", "--account", accountId, "--json"], accountId);
  validateEmptySecretList(await readStagingSecretList("staging_preexisting_secrets_forbidden", accountId));

  await runCommand("gh", ["auth", "status", "--hostname", "github.com"]);
  const userResult = await runCommand("gh", ["api", "/user"]);
  const user = parseJsonOutput(userResult.stdout, "github_cli_prerequisite_failed");
  if (user?.login !== CONTRACT.owner || user?.type !== CONTRACT.ownerType) safeFail("github_cli_prerequisite_failed");
  const repoResult = await runCommand("gh", ["api", `/repos/${CONTRACT.fullRepository}`]);
  const repo = parseJsonOutput(repoResult.stdout, "github_cli_prerequisite_failed");
  if (repo?.id !== CONTRACT.repositoryId || repo?.full_name !== CONTRACT.fullRepository
    || repo?.owner?.login !== CONTRACT.owner || repo?.owner?.type !== CONTRACT.ownerType
    || repo?.permissions?.admin !== true) safeFail("github_cli_prerequisite_failed");
  await verifyStoreOnlyHealth(stagingEndpoint);
}

function htmlEscape(value) {
  return value.replaceAll("&", "&amp;").replaceAll('"', "&quot;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");
}

export async function startManifestFlow() {
  const state = createManifestState(randomBytes);
  let manifest;
  let settled = false;
  let resolveCode;
  let rejectCode;
  let timer;
  const code = new Promise((resolve, reject) => { resolveCode = resolve; rejectCode = reject; });
  const server = createServer((request, response) => {
    const generic = (status, body) => {
      response.writeHead(status, {
        "content-type": "text/plain; charset=utf-8",
        "cache-control": "no-store",
        "x-content-type-options": "nosniff",
        "referrer-policy": "no-referrer",
      });
      response.end(body);
    };
    if (request.method === "GET" && request.url === CONTRACT.startPath && manifest) {
      const action = `https://github.com/settings/apps/new?state=${encodeURIComponent(state)}`;
      const body = `<!doctype html><html><head><meta charset="utf-8"><meta http-equiv="Content-Security-Policy" content="default-src 'none'; form-action https://github.com; style-src 'unsafe-inline'"><meta name="referrer" content="no-referrer"><title>Register private GitHub App</title></head><body><h1>Register the private results-ingest GitHub App</h1><p>Review the fixed personal-account settings, then continue. Do not edit the permissions or App name.</p><form action="${htmlEscape(action)}" method="post"><input type="hidden" name="manifest" value="${htmlEscape(JSON.stringify(manifest))}"><button type="submit">Continue to GitHub</button></form></body></html>`;
      response.writeHead(200, {
        "content-type": "text/html; charset=utf-8",
        "cache-control": "no-store",
        "content-security-policy": "default-src 'none'; form-action https://github.com; style-src 'unsafe-inline'",
        "referrer-policy": "no-referrer",
        "x-content-type-options": "nosniff",
      });
      response.end(body);
      return;
    }
    if (request.url?.startsWith(CONTRACT.callbackPath)) {
      if (settled) { generic(410, "This callback has already been consumed."); return; }
      try {
        const callbackCode = validateCallbackRequest({ method: request.method, requestUrl: request.url, expectedState: state });
        settled = true;
        generic(200, "GitHub App registration accepted. Return to the terminal to complete repository installation.");
        clearTimeout(timer);
        server.close();
        resolveCode(callbackCode);
      } catch {
        generic(400, "The GitHub App callback was rejected.");
      }
      return;
    }
    generic(404, "Not found.");
  });
  server.on("error", () => {
    if (!settled) { settled = true; rejectCode(new Error("github_app_loopback_failed")); }
  });
  await new Promise((resolve, reject) => {
    server.listen({ host: CONTRACT.loopbackHost, port: 0, exclusive: true }, resolve);
    server.once("error", () => reject(new Error("github_app_loopback_failed")));
  });
  const address = server.address();
  if (!address || typeof address === "string" || address.address !== CONTRACT.loopbackHost) {
    server.close();
    safeFail("github_app_loopback_failed");
  }
  const callbackUrl = `http://${CONTRACT.loopbackHost}:${address.port}${CONTRACT.callbackPath}`;
  manifest = buildGitHubAppManifest(callbackUrl);
  timer = setTimeout(() => {
    if (!settled) {
      settled = true;
      server.close();
      rejectCode(new Error("github_app_callback_timeout"));
    }
  }, CALLBACK_TIMEOUT_MS);
  return {
    startUrl: `http://${CONTRACT.loopbackHost}:${address.port}${CONTRACT.startPath}`,
    code,
    close: () => { clearTimeout(timer); server.close(); },
  };
}

export async function exchangeManifest(code) {
  if (typeof code !== "string" || !/^[A-Za-z0-9_-]{20,256}$/.test(code)) safeFail("github_app_conversion_failed");
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 30_000);
  let raw = "";
  try {
    const response = await fetch(`https://api.github.com/app-manifests/${encodeURIComponent(code)}/conversions`, {
      method: "POST",
      headers: {
        accept: "application/vnd.github+json",
        "content-type": "application/json",
        "user-agent": "cayleypy-results-ingest-bootstrap",
        "x-github-api-version": "2026-03-10",
      },
      redirect: "error",
      signal: controller.signal,
    });
    if (response.status !== 201) safeFail("github_app_conversion_failed");
    raw = await boundedText(response, 128 * 1024, "github_app_conversion_failed");
    return JSON.parse(raw);
  } catch { safeFail("github_app_conversion_failed"); }
  finally { raw = ""; clearTimeout(timer); }
}

async function githubApi(pathname, { bearer, method = "GET", body, acceptedStatuses = [200] }) {
  if (typeof pathname !== "string" || !pathname.startsWith("/") || pathname.startsWith("//")
    || typeof bearer !== "string" || bearer.length < 20 || bearer.length > 2048 || /\s/.test(bearer)) {
    safeFail("github_app_installation_lookup_failed");
  }
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 30_000);
  let raw = "";
  try {
    const response = await fetch(`https://api.github.com${pathname}`, {
      method,
      headers: {
        accept: "application/vnd.github+json",
        authorization: `Bearer ${bearer}`,
        ...(body === undefined ? {} : { "content-type": "application/json" }),
        "user-agent": "cayleypy-results-ingest-bootstrap",
        "x-github-api-version": "2026-03-10",
      },
      ...(body === undefined ? {} : { body: JSON.stringify(body) }),
      redirect: "error",
      signal: controller.signal,
    });
    if (!acceptedStatuses.includes(response.status)) safeFail("github_app_installation_lookup_failed");
    if (response.status === 204) return { body: undefined, link: response.headers.get("link") };
    raw = await boundedText(response, 256 * 1024, "github_app_installation_lookup_failed");
    return { body: JSON.parse(raw), link: response.headers.get("link") };
  } catch { safeFail("github_app_installation_lookup_failed"); }
  finally { raw = ""; clearTimeout(timer); }
}

function hasNextPage(link) {
  return typeof link === "string" && link.split(",").some((part) => /;\s*rel="?next"?\s*$/.test(part));
}

function exactReadOnlyTokenPermissions(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    && Object.keys(value).sort().join(",") === "contents,metadata"
    && value.contents === "read" && value.metadata === "read";
}

export async function waitForInstallation(appId, privateKeyPkcs8) {
  const deadline = Date.now() + INSTALL_TIMEOUT_MS;
  while (Date.now() < deadline) {
    let appJwt = createAppJwt({ appId, privateKeyPkcs8 });
    try {
      const listed = await githubApi("/app/installations?per_page=100", { bearer: appJwt });
      if (!Array.isArray(listed.body) || hasNextPage(listed.link)) safeFail("github_app_installation_invalid");
      if (listed.body.length === 0) {
        await new Promise((resolve) => setTimeout(resolve, 3_000));
        continue;
      }
      if (listed.body.length !== 1) safeFail("github_app_installation_invalid");
      const installation = listed.body[0];
      if (!Number.isSafeInteger(installation?.id) || installation.id <= 0) safeFail("github_app_installation_invalid");

      const repositoryInstallation = await githubApi(`/repos/${CONTRACT.fullRepository}/installation`, { bearer: appJwt });
      if (repositoryInstallation.body?.id !== installation.id) safeFail("github_app_installation_invalid");

      let installationToken = "";
      try {
        const tokenResponse = await githubApi(`/app/installations/${installation.id}/access_tokens`, {
          bearer: appJwt,
          method: "POST",
          body: { permissions: { contents: "read", metadata: "read" } },
          acceptedStatuses: [201],
        });
        const candidateToken = tokenResponse.body?.token;
        if (typeof candidateToken === "string" && candidateToken.length >= 20 && candidateToken.length <= 2048 && !/\s/.test(candidateToken)) {
          installationToken = candidateToken;
        }
        if (!installationToken || !exactReadOnlyTokenPermissions(tokenResponse.body?.permissions)
          || tokenResponse.body?.repository_selection !== "selected") safeFail("github_app_installation_invalid");
        const expiresAt = Date.parse(tokenResponse.body?.expires_at);
        if (!Number.isFinite(expiresAt) || expiresAt <= Date.now() || expiresAt > Date.now() + 3_700_000) {
          safeFail("github_app_installation_invalid");
        }
        tokenResponse.body.token = "";

        const repositories = await githubApi("/installation/repositories?per_page=100", { bearer: installationToken });
        if (hasNextPage(repositories.link)) safeFail("github_app_installation_invalid");
        const verified = verifyExactInstallation({ appId, installation, repositoryResponse: repositories.body });
        const repositoryVerified = verifyExactInstallation({
          appId,
          installation: repositoryInstallation.body,
          repositoryResponse: repositories.body,
        });
        if (repositoryVerified.installationId !== verified.installationId) safeFail("github_app_installation_invalid");
        return verified;
      } finally {
        if (installationToken) {
          await githubApi("/installation/token", {
            bearer: installationToken,
            method: "DELETE",
            acceptedStatuses: [204],
          });
          installationToken = "";
        }
      }
    } finally {
      appJwt = "";
    }
  }
  safeFail("github_app_installation_timeout");
}

export async function openBrowser(url) {
  let parsed;
  try { parsed = new URL(url); } catch { safeFail("browser_open_failed"); }
  const allowed = (parsed.protocol === "http:" && parsed.hostname === CONTRACT.loopbackHost)
    || (parsed.protocol === "https:" && parsed.hostname === "github.com");
  if (!allowed || parsed.username || parsed.password) safeFail("browser_open_failed");
  const target = parsed.href;
  const [command, args] = process.platform === "win32"
    ? ["rundll32.exe", ["url.dll,FileProtocolHandler", target]]
    : process.platform === "darwin" ? ["open", [target]] : ["xdg-open", [target]];
  await new Promise((resolve, reject) => {
    const child = spawn(command, args, { shell: false, windowsHide: true, detached: true, stdio: "ignore" });
    child.once("error", () => reject(new Error("browser_open_failed")));
    child.once("spawn", () => { child.unref(); resolve(); });
  });
}

export async function uploadSecrets({ appId, installationId, privateKeyPkcs8 }) {
  const { accountId } = await readAndValidateGeneratedConfig();
  validateEmptySecretList(await readStagingSecretList("staging_preexisting_secrets_forbidden", accountId));
  const invocation = buildSecretBulkInvocation({
    wranglerEntrypoint: WRANGLER_ENTRYPOINT,
    generatedConfigPath: GENERATED_CONFIG,
    appId,
    installationId,
    privateKeyPkcs8,
  });
  try {
    await runCommand(invocation.command, invocation.args, {
      stdin: invocation.stdin,
      capture: false,
      timeoutMs: 120_000,
      environment: exactWranglerEnvironment(accountId),
    });
    invocation.stdin = "";
    validateSecretList(await readStagingSecretList("secret_bulk_verification_failed", accountId));
  } catch (error) {
    if (error instanceof Error && error.message === "secret_bulk_verification_failed") throw error;
    safeFail("secret_bulk_failed");
  } finally {
    invocation.stdin = "";
  }
}

export function parseArguments(argv) {
  let dryRun = false;
  let confirmCreatePrivateApp = false;
  let stagingEndpoint;
  let help = false;
  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index];
    if (argument === "--dry-run") dryRun = true;
    else if (argument === "--confirm-create-private-app") confirmCreatePrivateApp = true;
    else if (argument === "--staging-endpoint" && index + 1 < argv.length) stagingEndpoint = argv[++index];
    else if (argument === "--help") help = true;
    else safeFail("invalid_bootstrap_arguments");
  }
  if (!help) {
    validateStagingEndpoint(stagingEndpoint);
    if (!dryRun && !confirmCreatePrivateApp) safeFail("explicit_live_confirmation_required");
  }
  return { dryRun, confirmCreatePrivateApp, stagingEndpoint, help };
}

export async function main(argv = process.argv.slice(2)) {
  const options = parseArguments(argv);
  if (options.help) {
    console.log("Usage: npm run bootstrap:github-app -- --staging-endpoint https://<staging-worker> [--dry-run | --confirm-create-private-app]");
    return;
  }
  const deps = { verifyPrerequisites, startManifestFlow, exchangeManifest, openBrowser, waitForInstallation, uploadSecrets, report: (message) => console.log(message) };
  await runBootstrap(options, deps);
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  main().catch((error) => {
    console.error(`BOOTSTRAP_FAILED: ${safeCode(error)}`);
    process.exitCode = 1;
  });
}
