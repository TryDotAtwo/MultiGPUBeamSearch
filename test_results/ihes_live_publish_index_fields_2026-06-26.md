# IHES live publish index field fix

Date: 2026-06-26

Scope:
- Fixed `hpc/ihes_cube_model/ihes_solve_bucket_from_scratch.sh` live publish index generation.
- The publisher now accepts existing per-puzzle metadata with `variants` and `source_files` keys from backfill output.
- Index and improvements writers filter rows to declared fields before CSV output.

Verification:
- `bash -n hpc/ihes_cube_model/ihes_solve_bucket_from_scratch.sh` in Docker passed.

Notes:
- The cluster log for puzzle 33 showed compute succeeded and local commit was created, but push failed because the compute node could not resolve `github.com`. That is an environment/network issue; local results remain in the results checkout and can be pushed later from a node with GitHub DNS/network.
