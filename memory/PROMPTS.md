2026-10-10 user requirement: Update existing public Kaggle notebook and leave only one GitHub branch in MultiGPUBeamSearch. Preserve checked source and historical reproducibility evidence. No additional GPU rentals.

## 2026-10-10 GNN requirements
Implement our own support for the two-level GNN architecture in issue #5, without using the submitted implementation as runtime. Provide both LibTorch and CUTLASS backends and preserve the five-stream search architecture.


GNN acceptance rental authorized: up to $5 total including disk/traffic. Implement our own GNN from issue5 with both LibTorch and CUTLASS; preserve beam architecture and do not use subagents.


2026-10-10: User requested running a large GNN comparison after the tiny fixture; approved a new RTX3060 rental up to $5 total. Compare matched Python/PyTorch, native LibTorch and native CUTLASS inference, archive results via GitHub and remove own rental.


## 2026-10-10: native GNN CUTLASS fusion
Requirement: improve large GNN native CUTLASS by at least 2x, preserving beam architecture and numerical correctness; profiling rental authorized up to $5.


## 2026-10-10: GNN fused rounded bias
User requested separate bias removal, RTX3060 speed/precision verification, an active goal and authorized $5 total for new tests.


## 2026-10-10: further GNN optimization with Astra
Continue improving GNN while measured gains remain, consult Astra as expert, preserve architecture; new GPU-test budget $10 total.

2026-10-11: Fix slow results publication and misleading published status. Keep the checked Cube555 solver and GitHub-only inputs; no new GPU execution.

## 2026-10-25 - Coordinated Cube555 notebook API candidate
- Integration owner authorized notebook/bootstrap/metadata/tests under beam-api-integration-20261025: preserve effective global beam 1048576, explicit NativeEnsemble, public graph.beam_search via enable_native, autotune=True, publication off; no commits/publication/GPU. Model/API bindings pending; candidate must fail closed without a published integration pin.
- Follow-up ownership instruction: finish local-prepared first-solution notebook using model-owner export_native_models and actual ordered puzzle_info/CSV graph; remove unsupported collect/reflection from this new example, keep baseline, execute synthetic CPU cells. Implemented and checked without GPU/publication.

## 2026-10-25 Cube555 model integration requirements
- Existing model owner assigned models.py/export.py, model-only helpers, artifact admission and parity tests on integration base1d2a1841. Export two explicit model families for NativeEnsemble, preserve150uint8 stickers and exact ordered graph/layout, FP16 body with explicit FP32 readout/blend; no new CUDA kernels. Native state/config/loader/API/autotuner/notebook changes owned by integrator/other owners. No commits/push/remote GPU/Molab restart. One tiny local synthetic CUDA parity allowed by existing coordinated work.

2026-10-25 integration continuation: finish public notebook -> CayleyPy -> explicit Cube555 blend -> existing autotuner/cache -> native search/replay on existing authorized resources. Preserve exact global beam and separate inference microbatch. Coordinate existing owners only; no new agents, rentals, publication or blocked Molab restart.

2026-10-25: Ivan authorized a cheap Vast environment via verified coordinator message to finish real Cube555 API/native/autotune/replay acceptance; no new agents/publication. Self-imposed capUSD2 and2h; preserve evidence then delete own lease only.

## 2026-10-25: hardware-aware complete setup requirement
User requirement relayed through existing coordinator and confirmed from human messages: prepare a puzzle on the actual GPUs; find the maximum safe frontier and profile grid; execute the width requested by the user using a measured valid profile. GPU type/free memory/count/topology and graph/model/runtime identity must participate. Verify full steps, cache reuse and width changes; do not substitute the previous 4096 smoke. Keep existing beam architecture, no new agents, paid acceptance remains within cumulative $2/2h cap.
- Later direct clarification (verified coordinator human messages): minimize bureaucracy and repeated checks. Maximum is a memory-admission diagnostic; full-step measurements select the profile at the actual requested width. Do not run every profile at the physical maximum merely for a report. Required end-to-end checks: actual logical width across saturated depths, replay where a solution exists, cache reuse and changed width. Preserve normal global selection architecture.

### 2026-10-25 — hardware-adaptive refinement continuation
Human requirements verified in coordinator thread 01a170d2-7c62-76e1-87ee-5668c52feb9e, turns 01a17474-1acd-7033-be0f-e244acc80452 and 01a17474-7936-73d1-b728-f05157d4fe41: generate the profile grid from actual GPU/VRAM/world/model/graph/beam, refine around the measured winner, reuse existing tuner and admission, preserve exact beam, bounded validation only. No global optimum claim or publication.

2026-10-25 direct user authorization: push the verified changes to GitHub and update the Kaggle notebook.

2026-10-25: User requests a clean Kaggle startup and end-to-end validation of the published notebook on two real T4 GPUs. Execute unchanged published cells/default parameters in a private diagnostic copy; GitHub-only source and model inputs.

2026-10-26: User says not to chase seconds/percent. Coordinator thread independently read: user requested useful profile heuristics/correlations and authorized Vast data collection. Implement bounded service-guided shortlist; initial self-imposed cap$2/2h, no new agents or140-step repetitions. Preserve current Kaggle validation.

2026-10-26 coordinator continuation: user requires maximum practical frontier without throughputcollapse, approximately linear scaling. Prior paired validation closed; investigate remaining allocation-only boundary. Only remaining firstpackage$2/2h total, no parallel lease, no longscramble/full sweep.

2026-10-26: User requires existing compatible profiles to be reused and persistent operation as in BFS; exact requested lifetime across separate calls is awaiting clarification. Avoid chasing tiny timing differences.

2026-10-26: User confirmed persistent GPU resources across separate beam_search calls and instructed completing implementation. Preserve architecture; reuse saved compatible profiles.

2026-10-26 follow-up via coordinator: finish working maximum near VRAM with full-step scaling evidence and CUTLASS persistent acceptance. No repeated successful LibTorch or full solve, no new agents, no budget increase. Use remaining cumulative2USD/2h or free Kaggle.

2026-10-26 continuation: finish the persistent cross-call implementation and validation; reuse compatible saved profiles before any new calibration; do not spend long runs optimizing tiny timing percentages. Preserve architecture and explicit beam width. No subagents.

2026-10-26: User authorized $10 total for continuation GPU tests (reply to 30-minute extension request). Initial new lease has30-minute protective stop, 2xRTX3060; no compilation on paid host.

2026-10-26: User requires calibration frontier to grow through ordinary native beam-search steps, not Python fixture generation. Only saturated full-frontier steps count.

2026-10-26: User approved up to$5 total for Cube555 Transformer+MLP0.8/0.2 working-maximum and full-step benchmark, following completed MLP-only validation.

2026-10-26: User requires portable Cube555 Transformer+MLP notebook using CayleyPy graph API, reusable outside Kaggle on an allocated GPU cluster node, automatic setup/profile selection, and GitHub-hosted previously verified hardware profiles (Kaggle profiles where evidence exists).

2026-10-26: Stream3 must reserve local and remote contributions before exchange; receiving without capacity must not be a normal waiting path. Preserve overlap and explicit requested beam.
`n2026-10-26: User requests cold Run All verification of published Cube555 blend notebook on Kaggle 2xT4, including setup and default search without manual intervention.

2026-10-26: User approved a human-readable notebook: imports, graph/models, beam, graph.beam_search, solution. Move download hashes and technical provenance out of visible example; retain cold Run All and portable setup. No subagents.

2026-10-27: User explicitly requests simplified notebook cold2xT4 test: natural growth to full frontier, ONE full step and stop, no140steps. Use actual full-input frontier evidence.
`n2026-10-27: Restore every editable old first-cell setting in readable CayleyPy notebook: model selection, collect, BFS radius, reflection and publish to cayleypy-beam-results. User wants the settings visible; hide only boilerplate.

2026-10-27 follow-up: User requests the downloaded cluster notebook retain every old first-cell setting (including model/checkpoint selection and blend weights) while using the concise CayleyPy graph.beam_search API. Expose collect, BFS radius/limit, reflection/source, puzzle selection, and publish replay-checked solutions through cayleypy-beam-results. Continue same scope on supported model; do not repeat the completed 2xT4 one-full-step acceptance or start a 140-step search. No subagents.


2026-10-27: User asked to finish the Cube555 notebook/API to full readiness. Keep all original first-cell model/search/collect/BFS/reflection/result-publication controls, preserve simple CayleyPy API, finish only bounded validation; do not repeat the accepted one-full-frontier-step run or start a 140-step search.


2026-10-28: Continue Cube555 notebook to end-to-end readiness: keep the first CayleyPy cell simple and expose all model/search/collect/BFS/reflection/publication options; use GitHub-pinned source; publish only replay-valid solutions through Cloudflare with verifiable Kaggle provenance; cold-check with bounded steps and no new rentals. User separately requested two 2.5M Kaggle examples (solution lengths ~10–12 and ~100); exact already-completed 12/103 runs were found, so do not duplicate them. Avoid 140-step acceptance runs.

2026-10-29: User requested evidence for portable frontier profiles on two affordable GPUs, using real Transformer+MLP blend (0.8/0.2), two beam widths, isolated Stream1 versus full-step cost, dispatch/job and kernel profiling, within the approved $2/90-minute lease. Scope excluded browser/Chrome extension, long search, 140 steps, and other instances.

2026-10-30: Close only the publication tail for the already-computed short/long Cube555 Kaggle results: do not run another search; correct long-result Kaggle provenance only if the existing receipt/API permits a safe correction; publish the replay-valid 12-move result only with exact provenance and idempotency; preserve receipts and state exact publication gaps.


2026-10-30: User reiterates the Cube555 blend notebook must be short and clear, use CayleyPy API without losing first-cell model/search/collect/BFS/reflection/GPU/cache/publish controls, and auto-select/reuse profiles like BFS; calibrate only when exact local/GitHub profile is absent.


2026-10-30 clarification: do not display a profile grid. Keep profile selection automatic inside the library. Make the Kaggle Cube555 notebook short, plainly readable, and free of visible validation/check code; retain all user model, search, collect, BFS, reflection, GPU and publishing controls.

2026-10-30 clarification: keep accepted GPU profiles in the GitHub source bundle (not only Kaggle-local cache); explain BEAM_MODE=simple, remove RUN_TIMEOUT_SECONDS, and add readable comments/spacing to the public Cube555 notebook.

2026-10-31 clarification: Keep `import multigpubeamsearch`. Remove the puzzle-specific `multigpubeamsearch.cube555` helper; configure the general workflow with a CayleyPy graph, supported predictor/model, and start state.

2026-10-30 clarification: publish the verified 2xT4 Cube555 profile to GitHub for the 1,048,576 beam; keep the public notebook readable with more comments, remove RUN_TIMEOUT_SECONDS, and explain simple beam mode.

2026-10-31: User wants the Cube555 notebook's public surface to be CayleyPy-only: no visible `multigpubeamsearch` or `cube555` imports. Expose native activation/session, puzzle/model loading, and replay/save/publish through CayleyPy or its official optional backend while retaining the native engine internally. The current mixed-import notebook does not meet this requirement.

2026-10-31 clarification: Keep the public `multigpubeamsearch` package import. Make this Kaggle notebook Cube555-specific and straightforward: remove generic graph/predictor branches and repeated validation code; preserve the first-cell settings for model weights/blend, beam search, collect, BFS, reflection and publication.
