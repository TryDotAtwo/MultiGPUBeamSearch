import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import { describe, expect, test } from "vitest";

type Binding = Record<string, unknown>;
type Environment = {
  durable_objects?: { bindings?: Binding[] };
  vars?: Record<string, string>;
  d1_databases?: Binding[];
  r2_buckets?: Binding[];
  queues?: { producers?: Binding[]; consumers?: Binding[] };
};

const configPath = fileURLToPath(new URL("../wrangler.jsonc", import.meta.url));
const config = JSON.parse(readFileSync(configPath, "utf8")) as {
  exports?: Record<string, unknown>;
  env?: Record<string, Environment>;
};

function bindingNames(values: Binding[] | undefined): string[] {
  return (values ?? []).map((value) => String(value.binding)).sort();
}

describe("Wrangler named-environment publication bindings", () => {
  test.each(["staging", "production"])("%s repeats every non-inheritable binding", (name) => {
    const value = config.env?.[name];
    expect(value).toBeDefined();
    expect(value?.durable_objects?.bindings).toEqual([
      { name: "GITHUB_WRITER", class_name: "GitHubWriter" },
    ]);
    expect(bindingNames(value?.d1_databases)).toEqual(["RESULTS_DB"]);
    expect(bindingNames(value?.r2_buckets)).toEqual(["RAW_RESULTS"]);
    expect(bindingNames(value?.queues?.producers)).toEqual(["VALIDATE_DLQ", "VALIDATE_QUEUE"]);
    expect(value?.queues?.consumers).toHaveLength(1);
  });

  test("staging and production storage and queues are isolated", () => {
    const staging = config.env?.staging;
    const production = config.env?.production;
    expect(staging?.d1_databases?.[0].database_name).not.toBe(production?.d1_databases?.[0].database_name);
    expect(staging?.r2_buckets?.[0].bucket_name).not.toBe(production?.r2_buckets?.[0].bucket_name);
    expect(staging?.queues?.consumers?.[0].queue).not.toBe(production?.queues?.consumers?.[0].queue);
  });

  test("new SQLite Durable Object export is top-level and legacy migrations are absent", () => {
    expect(config.exports).toEqual({
      GitHubWriter: { type: "durable-object", storage: "sqlite" },
    });
    expect(config).not.toHaveProperty("migrations");
  });

  test("deploy config contains no GitHub App secret or installation token", () => {
    const serialized = JSON.stringify(config);
    expect(serialized).not.toContain("GITHUB_APP_PRIVATE_KEY");
    expect(serialized).not.toContain("GITHUB_APP_INSTALLATION_TOKEN");
  });

  test("named environments start fail-closed until an explicit activation deployment", () => {
    expect(config.env?.staging?.vars?.INGEST_MODE).toBe("store_only");
    expect(config.env?.production?.vars?.INGEST_MODE).toBe("reject");
  });

  test("tracks a complete normal-mode config for Cloudflare Git builds", () => {
    const buildConfigPath = fileURLToPath(
      new URL("../wrangler.github-staging.jsonc", import.meta.url),
    );
    const buildConfig = JSON.parse(readFileSync(buildConfigPath, "utf8")) as {
      name?: string;
      account_id?: string;
      vars?: Record<string, string>;
      d1_databases?: Binding[];
    };
    expect(buildConfig.name).toBe("cayleypy-results-ingest-staging");
    expect(buildConfig.account_id).toBe("cdcad5aa88fc31bdb1f8508d2593ea88");
    expect(buildConfig.vars?.INGEST_MODE).toBe("normal");
    expect(buildConfig.d1_databases).toEqual([
      expect.objectContaining({
        binding: "RESULTS_DB",
        database_name: "cayleypy-results-staging",
        database_id: "0cbcdbb8-d382-451d-8940-fc33c761fef3",
        migrations_dir: "migrations",
      }),
    ]);
    expect(JSON.stringify(buildConfig)).not.toContain("GITHUB_APP_PRIVATE_KEY");
    const packageJson = JSON.parse(
      readFileSync(fileURLToPath(new URL("../package.json", import.meta.url)), "utf8"),
    ) as { scripts?: Record<string, string> };
    expect(packageJson.scripts?.["ci:cloudflare"]).toBe(
      "npm test && npm run typecheck",
    );
    expect(packageJson.scripts?.["deploy:staging:github"]).toBe(
      "wrangler deploy --config wrangler.github-staging.jsonc",
    );
  });});
