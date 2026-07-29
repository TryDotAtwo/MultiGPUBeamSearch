import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    include: ["test/schema.test.ts", "test/wrangler-config.test.ts", "test/load-recovery-gate.test.ts", "test/deployment-runbook.test.ts"],
    environment: "node",
  },
});
