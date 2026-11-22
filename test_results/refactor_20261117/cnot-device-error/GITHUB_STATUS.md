# CNOT descriptor recovery check — status

Date: 2026-11-17  
Previous source commit: `fe7de311d77918a0f0f213df5a41acaedd462b7d`

## Observed

- The two-rank descriptor-error command exited with status 1 at
  `DESCRIPTOR_ERROR_AND_REUSE`.
- The full failing log remains on the Vast instance; it was not copied to the
  Windows checkout and could not be read through the available GitHub run APIs.
  The exact assertion is therefore not confirmed from runtime output.

## Source-level diagnosis

- `NativeStages::initialize` places the initial parent frontier on rank 0.
  Hash ownership only chooses which rank stores the root in the closed set.
- For fixture rows `(8, 4, 2, 1)`, the source reference owner hash maps the
  root to rank 1 in a two-rank run.
- The previous test injected the invalid move only on that hash-owner rank.
  Rank 1 has an empty initial frontier, so it never called the score
  `argsort` that the injector wraps. The likely result is that no invalid move
  reached descriptor validation and the test raised
  `invalid generated move was not rejected`.

The test now injects on rank 0 and checks for a coordinated rank-0 error. This
is a source-derived diagnosis, not a confirmed explanation from the missing log.

## Acceptance state

Corrected two-rank rerun and same-process-group recovery are pending. No claim
is made that descriptor-error coordination or context reuse has passed.
