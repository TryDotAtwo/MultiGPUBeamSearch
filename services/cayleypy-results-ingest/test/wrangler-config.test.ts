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
});
