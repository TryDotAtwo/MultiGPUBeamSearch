declare module "node:crypto" {
  export interface Hash { update(data: string, encoding?: string): Hash; digest(encoding: "hex"): string; }
  export function createHash(name: "sha256"): Hash;
}

declare module "node:fs" {
  export function readFileSync(path: string | URL, encoding: "utf8"): string;
}

declare module "node:url" {
  export function fileURLToPath(url: string | URL): string;
}
