import { defineConfig } from "vitest/config";

export default defineConfig({ test: { include: ["test/schema.test.ts", "test/wrangler-config.test.ts"], environment: "node" } });
