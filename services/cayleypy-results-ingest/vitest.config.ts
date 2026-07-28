import { fileURLToPath } from "node:url";

import { defineWorkersConfig, readD1Migrations } from "@cloudflare/vitest-pool-workers/config";

export default defineWorkersConfig(async () => {
  const migrationsPath = fileURLToPath(new URL("./migrations", import.meta.url));
  const migrations = await readD1Migrations(migrationsPath);

  return {
    test: {
      include: ["test/receipt.test.ts"],
      setupFiles: ["./test/apply-migrations.ts"],
      poolOptions: {
        workers: {
          miniflare: {
            bindings: { TEST_MIGRATIONS: migrations },
            compatibilityDate: "2026-07-28",
            d1Databases: ["RESULTS_DB"],
            r2Buckets: ["RAW_RESULTS"],
          },
        },
      },
    },
  };
});
