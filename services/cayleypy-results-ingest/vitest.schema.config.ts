import { defineConfig } from "vitest/config";

export default defineConfig({ test: { include: ["test/schema.test.ts"], environment: "node" } });
