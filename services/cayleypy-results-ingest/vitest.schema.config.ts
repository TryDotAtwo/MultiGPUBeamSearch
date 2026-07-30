import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    include: [
      "test/schema.test.ts",
      "test/wrangler-config.test.ts",
      "test/load-recovery-gate.test.ts",
      "test/deployment-runbook.test.ts",
      "test/github-app-bootstrap.test.ts",
      "test/migration-upgrade.test.ts",
    ],
    environment: "node",
  },
});
