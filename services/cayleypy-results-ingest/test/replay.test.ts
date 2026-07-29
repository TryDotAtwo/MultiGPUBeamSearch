import { describe, expect, test } from "vitest";

import {
  invertPath,
  MAX_LOGICAL_STATE_LENGTH,
  MAX_MOVE_COUNT,
  MAX_PATH_LENGTH,
  replayPath,
} from "../src/replay.js";

describe("Task 4 bounded Cayley replay", () => {
  test("uses pull permutations for a deterministic composition", () => {
    const initial = [0, 1, 2, 3];
    const generators = { a: [1, 0, 3, 2], b: [2, 3, 0, 1] };
    expect(replayPath(initial, ["a", "b"], generators, 4)).toEqual({
      ok: true,
      state: [3, 2, 1, 0],
    });
  });

  test("composition agrees with independently expanded applications", () => {
    const initial = [0, 1, 2, 3, 4];
    const a = [4, 3, 2, 1, 0];
    const b = [1, 2, 3, 4, 0];
    const afterA = a.map((source) => initial[source]);
    const expected = b.map((source) => afterA[source]);
    expect(replayPath(initial, ["a", "b"], { a, b }, 5)).toEqual({
      ok: true,
      state: expected,
    });
  });

  test("rejects every out-of-bound logical state length", () => {
    expect(MAX_LOGICAL_STATE_LENGTH).toBe(120);
    expect(replayPath([0], [], {}, 0)).toEqual({
      ok: false,
      code: "state_length",
    });
    expect(replayPath([0], [], {}, 121)).toEqual({
      ok: false,
      code: "state_length",
    });
    expect(replayPath([0, 1], [], {}, 1)).toEqual({
      ok: false,
      code: "state_length",
    });
  });

  test("fails closed on path, move-count and proof-cell budgets", () => {
    const identity = [0];
    const tooManyMoves = Object.fromEntries(
      Array.from({ length: MAX_MOVE_COUNT + 1 }, (_, index) => [`m${index}`, identity]),
    );
    expect(
      replayPath(
        identity,
        Array.from({ length: MAX_PATH_LENGTH + 1 }, () => "m"),
        { m: identity },
        1,
      ),
    ).toEqual({ ok: false, code: "proof_bounds" });
    expect(replayPath(identity, [], tooManyMoves, 1)).toEqual({
      ok: false,
      code: "proof_bounds",
    });
  });

  test("rejects unknown and every malformed generator, including unused entries", () => {
    expect(replayPath([0, 1], ["missing"], { a: [0, 1] }, 2)).toEqual({
      ok: false,
      code: "unknown_move",
    });
    expect(
      replayPath([0, 1], ["good"], { good: [0, 1], unused_bad: [0, 0] }, 2),
    ).toEqual({ ok: false, code: "proof_bounds" });
    expect(replayPath([0, 1], ["bad"], { bad: [0, 2] }, 2)).toEqual({
      ok: false,
      code: "proof_bounds",
    });
  });

  test("fails closed when the inverse permutation has duplicate generator names", () => {
    expect(
      invertPath(
        ["forward"],
        {
          forward: [1, 0],
          backward_a: [1, 0],
          backward_b: [1, 0],
        },
        2,
      ),
    ).toEqual({ ok: false, code: "inverse_missing" });
  });
});
