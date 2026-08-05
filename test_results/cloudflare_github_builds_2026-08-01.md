# Cloudflare Workers Builds GitOps gate — 2026-08-01

- Repository: `TryDotAtwo/MultiGPUBeamSearch`
- Service root: `services/cayleypy-results-ingest`
- Worker: `cayleypy-results-ingest-staging`
- Config: `wrangler.github-staging.jsonc`
- Build: `npm ci && npm run ci:cloudflare`
- Deploy: `npm run deploy:staging:github`
- Runtime secrets remain encrypted in Cloudflare; none are present in GitHub.

Verification:

- TDD RED: tracked Git-build config absent;
- TDD RED: reproducible package scripts absent;
- GREEN: 57 schema/config tests passed;
- GREEN: 125 Worker tests passed;
- TypeScript typecheck passed;
- Wrangler deploy dry-run resolved the existing D1, R2, Queue, Durable Object, cron, and normal-mode bindings.

GitHub App installation is scoped only to `TryDotAtwo/MultiGPUBeamSearch`. Final dashboard connection requires GitHub sudo/2FA confirmation.