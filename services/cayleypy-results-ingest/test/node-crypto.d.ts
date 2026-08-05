declare module "node:crypto" {
  export interface Hash { update(data: string, encoding?: string): Hash; digest(encoding: "hex"): string; }
  export interface KeyObject {
    export(options: { type: "pkcs1" | "pkcs8"; format: "pem" }): { toString(): string };
  }
  export function createHash(name: "sha256"): Hash;
  export function generateKeyPairSync(type: "rsa", options: { modulusLength: number }): { privateKey: KeyObject };
}

declare module "node:fs" {
  export function readFileSync(path: string | URL, encoding: "utf8"): string;
}

declare module "node:url" {
  export function fileURLToPath(url: string | URL): string;
}
