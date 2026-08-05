import { SignJWT, importPKCS8 } from "jose";

export interface GitHubAppSecrets { GITHUB_APP_ID?: string; GITHUB_APP_INSTALLATION_ID?: string; GITHUB_APP_PRIVATE_KEY?: string; }
export interface GitHubAppConfig extends GitHubAppSecrets { REPO_OWNER?: string; REPO_NAME?: string; GITHUB_API_URL?: string; }
export interface GitHubFetch { (input: RequestInfo | URL, init?: RequestInit): Promise<Response>; }
export interface GitHubRequestOptions { method?: string; body?: unknown; token: string; }
export interface SafeGitHubResponse { status: number; body: unknown; }
interface CachedToken { value: string; refreshAt: number; }
interface TokenScope { appId: string; installationId: string; owner: string; repo: string; api: string; }

const JWT_MAX_AGE_SECONDS = 600;
const JWT_BACKDATE_SECONDS = 30;
const TOKEN_REFRESH_SKEW_MS = 60_000;
const INSTALLATION_TOKEN_PERMISSIONS = { contents: "write", metadata: "read" } as const;
const tokenCache = new Map<string, CachedToken>();
function unavailable(): Error { return new Error("github_app_auth_unavailable"); }
function failed(): Error { return new Error("github_app_auth_failed"); }
function required(value: string | undefined): string { if (!value || !value.trim()) throw unavailable(); return value.trim(); }
function numericId(value: string | undefined): string { const id = required(value); if (!/^\d+$/.test(id) || BigInt(id) <= 0n) throw unavailable(); return id; }
function apiUrl(config: GitHubAppConfig): string {
  const candidate = (config.GITHUB_API_URL ?? "https://api.github.com").trim().replace(/\/+$/, "");
  try { const parsed = new URL(candidate); if (parsed.protocol !== "https:" || parsed.username || parsed.password || parsed.search || parsed.hash) throw new Error(); return parsed.toString().replace(/\/+$/, ""); }
  catch { throw unavailable(); }
}
function scope(env: GitHubAppConfig): TokenScope { return { appId: numericId(env.GITHUB_APP_ID), installationId: numericId(env.GITHUB_APP_INSTALLATION_ID), owner: required(env.REPO_OWNER), repo: required(env.REPO_NAME), api: apiUrl(env) }; }
function cacheKey(value: TokenScope): string { return JSON.stringify([value.appId, value.installationId, value.owner, value.repo, value.api, INSTALLATION_TOKEN_PERMISSIONS]); }
function parseExpiry(value: unknown, now: number): number { if (typeof value !== "string") throw failed(); const expiry = Date.parse(value); if (!Number.isFinite(expiry) || expiry <= now) throw failed(); return expiry; }

/** Test-only reset; production methods never expose private keys or installation tokens. */
export function resetInstallationTokenCacheForTest(): void { tokenCache.clear(); }

/** Creates an RS256 JWT backdated for clock skew and valid for no more than ten minutes. */
export async function createAppJwt(env: GitHubAppSecrets, now = Date.now()): Promise<string> {
  try {
    const issuedAt = Math.floor(now / 1000) - JWT_BACKDATE_SECONDS;
    const key = await importPKCS8(required(env.GITHUB_APP_PRIVATE_KEY).replace(/\\n/g, "\n"), "RS256");
    return await new SignJWT({}).setProtectedHeader({ alg: "RS256", typ: "JWT" }).setIssuer(numericId(env.GITHUB_APP_ID)).setIssuedAt(issuedAt).setExpirationTime(issuedAt + JWT_MAX_AGE_SECONDS).sign(key);
  } catch (error) { if (error instanceof Error && error.message === "github_app_auth_unavailable") throw error; throw unavailable(); }
}

/** Obtains only a repository-scoped, least-privilege GitHub App installation token. */
export async function getInstallationToken(env: GitHubAppConfig, now = Date.now(), fetcher: GitHubFetch = fetch, beforeRequest?: () => void | Promise<void>): Promise<string> {
  const target = scope(env); const key = cacheKey(target); const cached = tokenCache.get(key);
  if (cached && now < cached.refreshAt) return cached.value;
  let jwt: string; try { jwt = await createAppJwt(env, now); } catch { throw unavailable(); }
  try { await beforeRequest?.(); } catch { throw failed(); }
  let response: Response;
  try {
    response = await fetcher(`${target.api}/app/installations/${encodeURIComponent(target.installationId)}/access_tokens`, { method: "POST", headers: { accept: "application/vnd.github+json", authorization: `Bearer ${jwt}`, "user-agent": "cayleypy-results-ingest", "x-github-api-version": "2026-03-10" }, body: JSON.stringify({ repositories: [target.repo], permissions: INSTALLATION_TOKEN_PERMISSIONS }) });
  } catch { throw failed(); }
  if (!response.ok) throw failed();
  let body: unknown; try { body = await response.json(); } catch { throw failed(); }
  if (body === null || typeof body !== "object") throw failed();
  const result = body as Record<string, unknown>;
  if (typeof result.token !== "string" || result.token.length === 0) throw failed();
  const expiry = parseExpiry(result.expires_at, now);
  tokenCache.set(key, { value: result.token, refreshAt: Math.max(now, expiry - TOKEN_REFRESH_SKEW_MS) });
  return result.token;
}

/** Safe Git data wrapper: error response bodies are never parsed or returned. */
export async function githubRequest(env: GitHubAppConfig, path: string, options: GitHubRequestOptions, fetcher: GitHubFetch = fetch): Promise<SafeGitHubResponse> {
  let api: string; try { api = apiUrl(env); if (!path.startsWith("/") || path.startsWith("//")) throw new Error(); } catch { throw new Error("github_temporary_unavailable"); }
  let response: Response;
  try { response = await fetcher(`${api}${path}`, { method: options.method ?? "GET", headers: { accept: "application/vnd.github+json", authorization: `Bearer ${options.token}`, "content-type": "application/json", "user-agent": "cayleypy-results-ingest", "x-github-api-version": "2026-03-10" }, body: options.body === undefined ? undefined : JSON.stringify(options.body) }); }
  catch { throw new Error("github_temporary_unavailable"); }
  if (!response.ok) return { status: response.status, body: undefined };
  try { return { status: response.status, body: await response.json() }; } catch { return { status: response.status, body: undefined }; }
}
