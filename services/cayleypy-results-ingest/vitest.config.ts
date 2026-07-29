import { fileURLToPath } from "node:url";

import { cloudflareTest, readD1Migrations } from "@cloudflare/vitest-pool-workers";
import { defineConfig } from "vitest/config";

export default defineConfig({
  plugins: [
    cloudflareTest(async () => {
      const migrationsPath = fileURLToPath(new URL("./migrations", import.meta.url));
      const migrations = await readD1Migrations(migrationsPath);

      return {
        miniflare: {
          bindings: { TEST_MIGRATIONS: migrations },
          compatibilityDate: "2026-07-28",
          d1Databases: ["RESULTS_DB"],
          r2Buckets: ["RAW_RESULTS"],
        },
      };
    }),
  ],
  test: {
    include: ["test/receipt.test.ts", "test/worker.test.ts", "test/consumer.test.ts", "test/replay.test.ts"],
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
