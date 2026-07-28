import { defineWorkersConfig } from "@cloudflare/vitest-pool-workers/config";

export default defineWorkersConfig({
  test: {
    include: ["test/receipt.test.ts"],
    poolOptions: {
      workers: { miniflare: { compatibilityDate: "2026-07-28", d1Databases: ["RESULTS_DB"], r2Buckets: ["RAW_RESULTS"] } },
    },
  },
});
