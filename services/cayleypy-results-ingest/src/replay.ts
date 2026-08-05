/** The native State128 payload has 120 logical bytes; padding is never proof data. */
export const MAX_LOGICAL_STATE_LENGTH = 120;
/** Bounded independently of JSON-schema validation for direct replay callers. */
export const MAX_PATH_LENGTH = 4096;
export const MAX_MOVE_COUNT = 256;
export const MAX_PROOF_INTEGER_CELLS = MAX_LOGICAL_STATE_LENGTH * (MAX_MOVE_COUNT + 2);

export type ProofState = readonly number[];
export type ProofGenerators = Readonly<Record<string, ProofState>>;

export type ReplayResult =
  | { ok: true; state: number[] }
  | { ok: false; code: "state_length" | "invalid_permutation" | "unknown_move" | "proof_bounds" };

function validState(state: ProofState, stateLength: number): boolean {
  return stateLength >= 1
    && stateLength <= MAX_LOGICAL_STATE_LENGTH
    && state.length === stateLength
    && state.every((value) => Number.isInteger(value) && value >= 0 && value < stateLength);
}

function validPermutation(permutation: ProofState, stateLength: number): boolean {
  if (!validState(permutation, stateLength)) return false;
  const seen = new Uint8Array(stateLength);
  for (const source of permutation) {
    if (!Number.isInteger(source) || source < 0 || source >= stateLength || seen[source] !== 0) return false;
    seen[source] = 1;
  }
  return true;
}

function validProofBounds(path: readonly string[], generators: ProofGenerators, stateLength: number): boolean {
  const entries = Object.entries(generators);
  if (path.length > MAX_PATH_LENGTH || entries.length > MAX_MOVE_COUNT) return false;
  // initial state plus central-state budget plus every generator; central is
  // not materialized here but reserving it keeps this pure helper fail-closed.
  if (stateLength * (entries.length + 2) > MAX_PROOF_INTEGER_CELLS) return false;
  return entries.every(([name, permutation]) => name.length > 0 && validPermutation(permutation, stateLength));
}

/**
 * Replays a named Cayley path using the same pull-permutation convention as
 * CayleyPy: next[i] = current[permutation[i]].
 */
export function replayPath(
  initial: ProofState,
  path: readonly string[],
  generators: ProofGenerators,
  stateLength: number,
): ReplayResult {
  if (!validState(initial, stateLength)) return { ok: false, code: "state_length" };
  if (!validProofBounds(path, generators, stateLength)) return { ok: false, code: "proof_bounds" };
  let state = [...initial];
  for (const token of path) {
    const permutation = generators[token];
    if (permutation === undefined) return { ok: false, code: "unknown_move" };
    state = permutation.map((source) => state[source]);
  }
  return { ok: true, state };
}
export type InvertPathResult =
  | { ok: true; path: string[] }
  | { ok: false; code: "proof_bounds" | "unknown_move" | "inverse_missing" };

function inversePermutation(permutation: ProofState): number[] {
  const inverse = new Array<number>(permutation.length);
  for (let target = 0; target < permutation.length; target += 1) inverse[permutation[target]] = target;
  return inverse;
}

/** Returns named inverse moves; reflection provenance must be expressible by the submitted generators. */
export function invertPath(path: readonly string[], generators: ProofGenerators, stateLength: number): InvertPathResult {
  if (!validProofBounds(path, generators, stateLength)) return { ok: false, code: "proof_bounds" };
  const namesByPermutation = new Map<string, string[]>();
  for (const [name, permutation] of Object.entries(generators)) {
    const key = permutation.join(",");
    const names = namesByPermutation.get(key) ?? [];
    names.push(name);
    namesByPermutation.set(key, names);
  }
  const inverted: string[] = [];
  for (let index = path.length - 1; index >= 0; index -= 1) {
    const permutation = generators[path[index]];
    if (permutation === undefined) return { ok: false, code: "unknown_move" };
    const inverseNames = namesByPermutation.get(inversePermutation(permutation).join(","));
    if (inverseNames === undefined || inverseNames.length !== 1) return { ok: false, code: "inverse_missing" };
    inverted.push(inverseNames[0]);
  }
  return { ok: true, path: inverted };
}
