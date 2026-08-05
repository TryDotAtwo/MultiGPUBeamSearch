import { fileURLToPath } from "node:url";

import { cloudflareTest, readD1Migrations } from "@cloudflare/vitest-pool-workers";
import { defineConfig } from "vitest/config";

export default defineConfig({
  plugins: [
    cloudflareTest(async () => {
      const migrationsPath = fileURLToPath(new URL("./migrations", import.meta.url));
      const migrations = await readD1Migrations(migrationsPath);

      return {
        main: "./src/worker.ts",
        miniflare: {
          bindings: {
            TEST_MIGRATIONS: migrations,
            INGEST_MODE: "normal",
            GITHUB_APP_ID: "1",
            GITHUB_APP_INSTALLATION_ID: "1",
            GITHUB_APP_PRIVATE_KEY: "test-only-placeholder",
            GITHUB_API_URL: "https://github.example/api/v3",
            REPO_OWNER: "owner",
            REPO_NAME: "repo",
            STAGING_BRANCH: "ingest/staging",
          },
          compatibilityDate: "2026-07-28",
          d1Databases: ["RESULTS_DB"],
          r2Buckets: ["RAW_RESULTS"],
          durableObjects: { GITHUB_WRITER: "GitHubWriter" },
        },
      };
    }),
  ],
  test: {
    include: ["test/receipt.test.ts", "test/worker-v2.test.ts", "test/worker.test.ts", "test/consumer.test.ts", "test/replay.test.ts", "test/github-app.test.ts", "test/github-writer.test.ts"],
    deps: {
      optimizer: {
        ssr: {
          enabled: true,
          include: ["ajv/dist/2020.js", "ajv-formats"],
        },
      },
    },
    setupFiles: ["./test/apply-migrations.ts"],
  },
});
