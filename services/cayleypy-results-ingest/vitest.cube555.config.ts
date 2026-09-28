import { defineConfig } from "vitest/config";
export default defineConfig({ test: { include: ["test/schema.test.ts", "test/schema-v2.test.ts", "test/replay.test.ts", "test/cube555.test.ts"], environment: "node" }});
