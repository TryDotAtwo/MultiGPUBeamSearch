export interface BootstrapContract {
  appName: string; loopbackHost: string; callbackPath: string; startPath: string;
  wranglerVersion: string; environment: string; workerName: string;
  owner: string; ownerType: string; repository: string; fullRepository: string;
  repositoryId: number; stagingBranch: string; d1Name: string; r2Name: string;
  queueName: string; dlqName: string;
}
export const CONTRACT: Readonly<BootstrapContract>;
export function createManifestState(randomBytes: (length: number) => Uint8Array): string;
export function buildGitHubAppManifest(callbackUrl: string): Record<string, unknown>;
export function validateCallbackRequest(input: { method?: string; requestUrl?: string; expectedState: string }): string;
export function validateCloudflareAccountOverride(value: string | undefined, expectedAccountId: string): string;
export function buildExactCloudflareEnvironment(environment: Record<string, string | undefined>, expectedAccountId: string): Record<string, string | undefined>;
export function convertPkcs1ToPkcs8(privateKeyPkcs1: string): string;
export function createAppJwt(input: { appId: number; privateKeyPkcs8: string; nowMs?: number }): string;
export function validateGeneratedStoreOnlyConfig(input: { config: unknown; configPath: string; serviceRoot: string }): { accountId: string; d1DatabaseId: string };
export function validateConversion(value: unknown): { appId: number; privateKeyPkcs1: string; installUrl: string };
export function verifyExactInstallation(input: {
  appId: number;
  installation: unknown;
  repositoryResponse: unknown;
}): { installationId: number };
export function buildSecretBulkInvocation(input: {
  wranglerEntrypoint: string; generatedConfigPath: string; appId: number;
  installationId: number; privateKeyPkcs8: string;
}): { command: string; args: string[]; stdin: string };
export function validateEmptySecretList(value: unknown): [];
export function validateSecretList(value: unknown): string[];
export function validateStagingEndpoint(value: string): string;
export interface BootstrapOptions { dryRun: boolean; stagingEndpoint: string; confirmCreatePrivateApp?: boolean; }
export interface BootstrapDependencies {
  verifyPrerequisites(input: { stagingEndpoint: string; finalCheck: boolean }): Promise<void>;
  startManifestFlow(): Promise<{ startUrl: string; code: Promise<string>; close?: () => void }>;
  exchangeManifest(code: string): Promise<unknown>;
  openBrowser(url: string): Promise<void>;
  waitForInstallation(appId: number, privateKeyPkcs8: string): Promise<{ installationId: number }>;
  uploadSecrets(input: { appId: number; installationId: number; privateKeyPkcs8: string }): Promise<void>;
  report(message: string): void;
}
export function runBootstrap(options: BootstrapOptions, deps: BootstrapDependencies): Promise<
  { status: "dry_run_ok" } | { status: "ok"; appId: number; installationId: number }
>;
export function startManifestFlow(): Promise<{ startUrl: string; code: Promise<string>; close(): void }>;
export function exchangeManifest(code: string): Promise<unknown>;
export function waitForInstallation(appId: number, privateKeyPkcs8: string): Promise<{ installationId: number }>;
export function openBrowser(url: string): Promise<void>;
export function uploadSecrets(input: { appId: number; installationId: number; privateKeyPkcs8: string }): Promise<void>;
export type ParsedArguments =
  | { help: true; dryRun: boolean; confirmCreatePrivateApp: boolean; stagingEndpoint: undefined }
  | { help: false; dryRun: boolean; confirmCreatePrivateApp: boolean; stagingEndpoint: string };
export function parseArguments(argv: string[]): ParsedArguments;
export function main(argv?: string[]): Promise<void>;
export function runCommand(command: string, args: string[], options?: { stdin?: string; capture?: boolean; timeoutMs?: number; environment?: Record<string, string | undefined> }): Promise<{ stdout: string }>;
