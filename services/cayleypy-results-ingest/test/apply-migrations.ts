import { applyD1Migrations, env, type D1Migration } from "cloudflare:test";

import type { IngestEnv } from "../src/storage.js";

declare module "cloudflare:test" {
  interface ProvidedEnv extends Pick<IngestEnv, "RESULTS_DB" | "RAW_RESULTS"> {
    TEST_MIGRATIONS: D1Migration[];
  }
}

await applyD1Migrations(env.RESULTS_DB, env.TEST_MIGRATIONS);
