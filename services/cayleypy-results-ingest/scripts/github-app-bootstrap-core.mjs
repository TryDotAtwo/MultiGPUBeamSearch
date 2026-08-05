import { createPrivateKey, sign, timingSafeEqual } from "node:crypto";
import path from "node:path";

export const CONTRACT = Object.freeze({
  appName: "cayleypy-beam-results-ingest",
  loopbackHost: "127.0.0.1",
  callbackPath: "/github-app-manifest/callback",
  startPath: "/github-app-manifest/start",
  wranglerVersion: "4.115.0",
  environment: "staging",
  workerName: "cayleypy-results-ingest",
  owner: "TryDotAtwo",
  ownerType: "User",
  repository: "cayleypy-beam-results",
  fullRepository: "TryDotAtwo/cayleypy-beam-results",
  repositoryId: 1_281_329_788,
  stagingBranch: "ingest/staging",
  d1Name: "cayleypy-results-staging",
  r2Name: "cayleypy-results-raw-staging",
  queueName: "cayleypy-validate-staging",
  dlqName: "cayleypy-validate-dlq-staging",
});

const SAFE_PERMISSIONS = Object.freeze({ contents: "write", metadata: "read" });

function fail(code) { throw new Error(code); }
function isObject(value) { return value !== null && typeof value === "object" && !Array.isArray(value); }
function exactObject(value, expected) {
  if (!isObject(value)) return false;
  const keys = Object.keys(value).sort();
  const expectedKeys = Object.keys(expected).sort();
  return keys.length === expectedKeys.length
    && keys.every((key, index) => key === expectedKeys[index])
    && keys.every((key) => value[key] === expected[key]);
}
function pathApiFor(value) { return /^[A-Za-z]:[\\/]/.test(value) ? path.win32 : path; }
function samePath(left, right, api) {
  const a = api.resolve(left);
  const b = api.resolve(right);
  return api === path.win32 ? a.toLowerCase() === b.toLowerCase() : a === b;
}

export function createManifestState(randomBytes) {
  let bytes;
  try { bytes = randomBytes(32); } catch { fail("github_app_state_generation_failed"); }
  if (!(bytes instanceof Uint8Array) || bytes.byteLength !== 32) fail("github_app_state_generation_failed");
  const state = Buffer.from(bytes).toString("base64url");
  if (!/^[A-Za-z0-9_-]{43}$/.test(state)) fail("github_app_state_generation_failed");
  return state;
}

export function buildGitHubAppManifest(callbackUrl) {
  let callback;
  try { callback = new URL(callbackUrl); } catch { fail("github_app_callback_url_invalid"); }
  if (callback.protocol !== "http:" || callback.hostname !== CONTRACT.loopbackHost
    || !callback.port || callback.pathname !== CONTRACT.callbackPath
    || callback.username || callback.password || callback.search || callback.hash) {
    fail("github_app_callback_url_invalid");
  }
  return {
    name: CONTRACT.appName,
    url: `https://github.com/${CONTRACT.fullRepository}`,
    redirect_url: callback.href,
    hook_attributes: { url: `https://github.com/${CONTRACT.fullRepository}`, active: false },
    public: false,
    default_permissions: { ...SAFE_PERMISSIONS },
    default_events: [],
    request_oauth_on_install: false,
    setup_on_update: false,
  };
}

export function validateCallbackRequest({ method, requestUrl, expectedState }) {
  try {
    if (method !== "GET" || typeof requestUrl !== "string" || !requestUrl.startsWith("/") || requestUrl.startsWith("//")) fail("github_app_callback_rejected");
    const parsed = new URL(requestUrl, `http://${CONTRACT.loopbackHost}`);
    if (parsed.pathname !== CONTRACT.callbackPath || parsed.hash) fail("github_app_callback_rejected");
    const keys = [...parsed.searchParams.keys()].sort();
    if (keys.length !== 2 || keys[0] !== "code" || keys[1] !== "state") fail("github_app_callback_rejected");
    const codes = parsed.searchParams.getAll("code");
    const states = parsed.searchParams.getAll("state");
    if (codes.length !== 1 || states.length !== 1 || !/^[A-Za-z0-9_-]{20,256}$/.test(codes[0])) fail("github_app_callback_rejected");
    if (!/^[A-Za-z0-9_-]{43}$/.test(expectedState) || states[0].length !== expectedState.length) fail("github_app_callback_rejected");
    if (!timingSafeEqual(Buffer.from(states[0]), Buffer.from(expectedState))) fail("github_app_callback_rejected");
    return codes[0];
  } catch { fail("github_app_callback_rejected"); }
}

export function convertPkcs1ToPkcs8(privateKeyPkcs1) {
  try {
    if (typeof privateKeyPkcs1 !== "string"
      || !privateKeyPkcs1.startsWith("-----BEGIN RSA PRIVATE KEY-----\n")
      || !privateKeyPkcs1.trimEnd().endsWith("-----END RSA PRIVATE KEY-----")) fail("github_app_private_key_invalid");
    const key = createPrivateKey(privateKeyPkcs1);
    const exported = key.export({ type: "pkcs8", format: "pem" });
    const privateKeyPkcs8 = typeof exported === "string" ? exported : exported.toString("utf8");
    if (!privateKeyPkcs8.startsWith("-----BEGIN PRIVATE KEY-----\n")) fail("github_app_private_key_invalid");
    createPrivateKey(privateKeyPkcs8);
    return privateKeyPkcs8;
  } catch { fail("github_app_private_key_invalid"); }
}

export function createAppJwt({ appId, privateKeyPkcs8, nowMs = Date.now() }) {
  try {
    if (!Number.isSafeInteger(appId) || appId <= 0 || !Number.isFinite(nowMs)
      || typeof privateKeyPkcs8 !== "string" || !privateKeyPkcs8.startsWith("-----BEGIN PRIVATE KEY-----\n")) {
      fail("github_app_jwt_failed");
    }
    const nowSeconds = Math.floor(nowMs / 1000);
    const header = Buffer.from(JSON.stringify({ alg: "RS256", typ: "JWT" })).toString("base64url");
    const payload = Buffer.from(JSON.stringify({ iat: nowSeconds - 30, exp: nowSeconds + 540, iss: String(appId) })).toString("base64url");
    const signingInput = `${header}.${payload}`;
    const signature = sign("RSA-SHA256", Buffer.from(signingInput), createPrivateKey(privateKeyPkcs8)).toString("base64url");
    return `${signingInput}.${signature}`;
  } catch { fail("github_app_jwt_failed"); }
}

export function validateGeneratedStoreOnlyConfig({ config, configPath, serviceRoot }) {
  try {
    if (!isObject(config) || typeof configPath !== "string" || typeof serviceRoot !== "string") fail("staging_secret_sink_invalid");
    const api = pathApiFor(serviceRoot);
    const expectedConfig = api.join(serviceRoot, ".staging-deploy-private", "wrangler.generated.json");
    if (!samePath(configPath, expectedConfig, api)) fail("staging_secret_sink_invalid");
    if (config.name !== CONTRACT.workerName
      || typeof config.account_id !== "string"
      || !/^[0-9a-f]{32}$/.test(config.account_id)
      || Object.hasOwn(config, "legacy_env")) fail("staging_secret_sink_invalid");
    const staging = config.env?.staging;
    if (!isObject(staging) || Object.hasOwn(staging, "name") || Object.hasOwn(staging, "account_id")
      || Object.hasOwn(staging, "legacy_env")) fail("staging_secret_sink_invalid");
    const vars = staging.vars;
    if (!exactObject(vars, {
      INGEST_MODE: "store_only",
      REPO_OWNER: CONTRACT.owner,
      REPO_NAME: CONTRACT.repository,
      STAGING_BRANCH: CONTRACT.stagingBranch,
    })) fail("staging_secret_sink_invalid");

    const d1 = staging.d1_databases;
    if (!Array.isArray(d1) || d1.length !== 1 || d1[0]?.binding !== "RESULTS_DB" || d1[0]?.database_name !== CONTRACT.d1Name
      || typeof d1[0]?.database_id !== "string"
      || !/^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$/.test(d1[0].database_id)
      || typeof d1[0]?.migrations_dir !== "string"
      || !samePath(d1[0].migrations_dir, api.join(serviceRoot, "migrations"), api)) fail("staging_secret_sink_invalid");
    const r2 = staging.r2_buckets;
    if (!Array.isArray(r2) || r2.length !== 1 || !exactObject(r2[0], { binding: "RAW_RESULTS", bucket_name: CONTRACT.r2Name })) fail("staging_secret_sink_invalid");
    const producers = staging.queues?.producers;
    const consumers = staging.queues?.consumers;
    if (!Array.isArray(producers) || producers.length !== 2
      || !exactObject(producers[0], { binding: "VALIDATE_QUEUE", queue: CONTRACT.queueName })
      || !exactObject(producers[1], { binding: "VALIDATE_DLQ", queue: CONTRACT.dlqName })
      || !Array.isArray(consumers) || consumers.length !== 1
      || !exactObject(consumers[0], { queue: CONTRACT.queueName, max_batch_size: 10, max_retries: 8, dead_letter_queue: CONTRACT.dlqName })) fail("staging_secret_sink_invalid");
    const serialized = JSON.stringify(config);
    if (/GITHUB_APP_(?:ID|INSTALLATION_ID|PRIVATE_KEY)/.test(serialized) || /BEGIN (?:RSA )?PRIVATE KEY/.test(serialized)) fail("staging_secret_sink_invalid");
    return { accountId: config.account_id, d1DatabaseId: d1[0].database_id };
  } catch { fail("staging_secret_sink_invalid"); }
}

export function validateCloudflareAccountOverride(value, expectedAccountId) {
  try {
    if (typeof expectedAccountId !== "string" || !/^[0-9a-f]{32}$/.test(expectedAccountId)
      || (value !== undefined && (typeof value !== "string" || value.toLowerCase() !== expectedAccountId))) {
      fail("cloudflare_account_mismatch");
    }
    return expectedAccountId;
  } catch { fail("cloudflare_account_mismatch"); }
}

export function buildExactCloudflareEnvironment(environment, expectedAccountId) {
  try {
    if (!isObject(environment)) fail("cloudflare_account_mismatch");
    validateCloudflareAccountOverride(undefined, expectedAccountId);
    const sanitized = { ...environment };
    for (const key of Object.keys(sanitized)) {
      if (key.toUpperCase() === "CLOUDFLARE_ACCOUNT_ID") {
        validateCloudflareAccountOverride(sanitized[key], expectedAccountId);
        delete sanitized[key];
      }
    }
    sanitized.CLOUDFLARE_ACCOUNT_ID = expectedAccountId;
    return sanitized;
  } catch { fail("cloudflare_account_mismatch"); }
}

export function validateConversion(value) {
  try {
    if (!isObject(value) || !Number.isSafeInteger(value.id) || value.id <= 0
      || value.name !== CONTRACT.appName || value.public !== false
      || value.owner?.login !== CONTRACT.owner || value.owner?.type !== CONTRACT.ownerType
      || !exactObject(value.permissions, SAFE_PERMISSIONS)
      || !Array.isArray(value.events) || value.events.length !== 0
      || typeof value.pem !== "string" || !value.pem.startsWith("-----BEGIN RSA PRIVATE KEY-----\n")) fail("github_app_conversion_invalid");
    const htmlUrl = new URL(value.html_url);
    const expectedAppPath = `/apps/${CONTRACT.appName}`;
    if (htmlUrl.protocol !== "https:" || htmlUrl.hostname !== "github.com"
      || htmlUrl.pathname !== expectedAppPath
      || htmlUrl.search || htmlUrl.hash || htmlUrl.username || htmlUrl.password) fail("github_app_conversion_invalid");
    return { appId: value.id, privateKeyPkcs1: value.pem, installUrl: `https://github.com${expectedAppPath}/installations/new` };
  } catch { fail("github_app_conversion_invalid"); }
}

export function verifyExactInstallation({ appId, installation, repositoryResponse }) {
  try {
    if (!Number.isSafeInteger(appId) || appId <= 0 || !isObject(installation) || !isObject(repositoryResponse)
      || installation.app_id !== appId || !Number.isSafeInteger(installation.id) || installation.id <= 0
      || installation.account?.login !== CONTRACT.owner || installation.account?.type !== CONTRACT.ownerType
      || installation.repository_selection !== "selected" || installation.suspended_at !== null
      || !exactObject(installation.permissions, SAFE_PERMISSIONS)
      || !Array.isArray(installation.events) || installation.events.length !== 0) fail("github_app_installation_invalid");
    const repositories = repositoryResponse.repositories;
    if (repositoryResponse.total_count !== 1 || !Array.isArray(repositories) || repositories.length !== 1
      || repositories[0]?.id !== CONTRACT.repositoryId || repositories[0]?.full_name !== CONTRACT.fullRepository) fail("github_app_installation_invalid");
    return { installationId: installation.id };
  } catch { fail("github_app_installation_invalid"); }
}

export function buildSecretBulkInvocation({ wranglerEntrypoint, generatedConfigPath, appId, installationId, privateKeyPkcs8 }) {
  if (typeof wranglerEntrypoint !== "string" || !wranglerEntrypoint || typeof generatedConfigPath !== "string"
    || !Number.isSafeInteger(appId) || appId <= 0 || !Number.isSafeInteger(installationId) || installationId <= 0
    || typeof privateKeyPkcs8 !== "string" || !privateKeyPkcs8.startsWith("-----BEGIN PRIVATE KEY-----\n")) {
    fail("secret_bulk_input_invalid");
  }
  return {
    command: process.execPath,
    args: [wranglerEntrypoint, "secret", "bulk", "--config", generatedConfigPath, "--env", CONTRACT.environment],
    stdin: `${JSON.stringify({
      GITHUB_APP_ID: String(appId),
      GITHUB_APP_INSTALLATION_ID: String(installationId),
      GITHUB_APP_PRIVATE_KEY: privateKeyPkcs8,
    })}\n`,
  };
}

export function validateEmptySecretList(value) {
  try {
    if (!Array.isArray(value) || value.length !== 0) fail("staging_preexisting_secrets_forbidden");
    return [];
  } catch { fail("staging_preexisting_secrets_forbidden"); }
}

export function validateSecretList(value) {
  const expected = ["GITHUB_APP_ID", "GITHUB_APP_INSTALLATION_ID", "GITHUB_APP_PRIVATE_KEY"].sort();
  try {
    if (!Array.isArray(value) || value.length !== expected.length) fail("secret_bulk_verification_failed");
    const actual = value.map((entry) => isObject(entry) && entry.type === "secret_text" && typeof entry.name === "string" ? entry.name : "").sort();
    if (actual.some((name, index) => name !== expected[index])) fail("secret_bulk_verification_failed");
    return actual;
  } catch { fail("secret_bulk_verification_failed"); }
}
export function validateStagingEndpoint(value) {
  try {
    const endpoint = new URL(value);
    if (endpoint.protocol !== "https:" || endpoint.username || endpoint.password || endpoint.search || endpoint.hash
      || (endpoint.pathname !== "/" && endpoint.pathname !== "")) fail("staging_endpoint_invalid");
    return endpoint.origin;
  } catch { fail("staging_endpoint_invalid"); }
}

export async function runBootstrap(options, deps) {
  const stagingEndpoint = validateStagingEndpoint(options?.stagingEndpoint);
  await deps.verifyPrerequisites({ stagingEndpoint, finalCheck: false });
  deps.report("PREFLIGHT_OK: exact read-only prerequisites verified");
  if (options?.dryRun === true) {
    deps.report("DRY_RUN_OK: no App, resource, browser flow, or secret mutation was attempted");
    return { status: "dry_run_ok" };
  }
  if (options?.confirmCreatePrivateApp !== true) fail("explicit_live_confirmation_required");
  let rawConversion;
  let privateKeyPkcs1 = "";
  let privateKeyPkcs8 = "";
  let flow;
  try {
    flow = await deps.startManifestFlow();
    await deps.openBrowser(flow.startUrl);
    deps.report("MANIFEST_FLOW_READY: complete the private App form in the opened browser");
    const code = await flow.code;
    rawConversion = await deps.exchangeManifest(code);
    const conversion = validateConversion(rawConversion);
    privateKeyPkcs1 = conversion.privateKeyPkcs1;
    if (isObject(rawConversion)) {
      rawConversion.pem = "";
      if (typeof rawConversion.client_secret === "string") rawConversion.client_secret = "";
      if (typeof rawConversion.webhook_secret === "string") rawConversion.webhook_secret = "";
    }
    rawConversion = undefined;
    privateKeyPkcs8 = convertPkcs1ToPkcs8(privateKeyPkcs1);
    privateKeyPkcs1 = "";
    conversion.privateKeyPkcs1 = "";
    await deps.openBrowser(conversion.installUrl);
    deps.report("INSTALL_FLOW_READY: choose Only select repositories and the one required repository");
    const { installationId } = await deps.waitForInstallation(conversion.appId, privateKeyPkcs8);
    await deps.verifyPrerequisites({ stagingEndpoint, finalCheck: true });
    await deps.uploadSecrets({ appId: conversion.appId, installationId, privateKeyPkcs8 });
    deps.report("BOOTSTRAP_OK: exact staging secret names installed through one bulk version");
    return { status: "ok", appId: conversion.appId, installationId };
  } finally {
    if (isObject(rawConversion)) {
      rawConversion.pem = "";
      if (typeof rawConversion.client_secret === "string") rawConversion.client_secret = "";
      if (typeof rawConversion.webhook_secret === "string") rawConversion.webhook_secret = "";
    }
    flow?.close?.();
    privateKeyPkcs1 = "";
    privateKeyPkcs8 = "";
  }
}
