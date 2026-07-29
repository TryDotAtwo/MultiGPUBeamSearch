declare module "node:crypto" {
  export interface Hash { update(data: string, encoding?: string): Hash; digest(encoding: "hex"): string; }
  export function createHash(name: "sha256"): Hash;
}
