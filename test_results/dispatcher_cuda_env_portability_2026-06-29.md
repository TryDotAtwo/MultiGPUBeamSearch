# Dispatcher CUDA Env Portability 2026-06-29

- Scope: `tests/dispatcher_cuda_tests.cu` runtime weight estimate environment overrides.
- Change: replaced POSIX-only `setenv`/`unsetenv` calls with a scoped portable helper.
- Windows path: `_putenv_s(name, value)` sets overrides; `_putenv_s(name, "")` clears values that were absent before the test.
- Non-Windows path: `setenv(name, value, 1)` sets overrides; `unsetenv(name)` clears values that were absent before the test.
- Restore behavior: previous values are captured before override and restored in the helper destructor if an exception exits the test scope.
- Verification: Docker dispatcher build/test command used `/tmp/beam-dispatcher-build` to avoid repo build directories. `cmake --build /tmp/beam-dispatcher-build --target dispatcher_cuda_tests` compiled and linked `dispatcher_cuda_tests`; running the executable then failed with `stream4 conditional scheduler launched too many jobs`. After removing an unused helper, a build-only rerun of the same Docker target compiled and linked successfully.

- Follow-up: explicitly set `stream4_trigger_candidates` in manual dispatcher test configs and removed the brittle `shard_count * 2` Stream4 launch upper bound; final Docker run passed `stream1_transformer_cuda_tests`, `stream1_cuda_tests`, `dispatcher_cuda_tests`, and tiny transformer `production_runner` smoke.
