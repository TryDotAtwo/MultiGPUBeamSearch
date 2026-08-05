import { env } from "cloudflare:workers";
import { applyD1Migrations, type D1Migration } from "cloudflare:test";

import type { IngestEnv } from "../src/storage.js";

declare global {
  namespace Cloudflare {
    interface Env extends Pick<IngestEnv, "RESULTS_DB" | "RAW_RESULTS"> {
      GITHUB_WRITER: DurableObjectNamespace<import("../src/github-writer.js").GitHubWriter>;
      TEST_MIGRATIONS: D1Migration[];
    }
  }
}

await applyD1Migrations(env.RESULTS_DB, env.TEST_MIGRATIONS);
