# Service-guided shortlist implementation plan

Goal: reduce profile-selection work while preserving inference-first/exact-width contracts.
Architecture: keep existing native planning, component measurements and full-step selection. Rank a small candidate set from measured service pressure; no CUDA changes.

Evidence: 4060Ti/65536 baseline downstream work .036s versus inference5.63s; 2T4/1M .537s versus84.13s. Small service flush worsened sort work to .446s and35.95s. These isolated measurements are not overlap or full-step guarantees.

- [ ] Add pure shortlist policy using baseline service ratios and candidate geometry, cap at four probes. Low-pressure case retains independent outer/ring/lanes/flush probes; do not blindly shrink flush. Preserve inference microbatch and exact width.
- [ ] Full-step baseline plus one challenger when baseline downstream/inference<=.05; otherwise up to two. Keep complete timing cohorts, correctness gates, and fallback. Stop refinement if coarse round finds no statistically separated full-step gain.
- [ ] Bind versioned policy to cache, retain explicit trace and incomplete-failure exclusion.
- [ ] Check existing held-out A4000/T4 records and tests; then one cheap GPU for new exact-width run and replay. No repeat140-depth solve.
- [ ] Rental: sole owner, max$2 total first package, max2h, watchdog, archive verification before destroy. Current Kaggle v3 remains separate and untouched.

Alternatives: hardware-only large grid is current expensive baseline; pure predicted winner cannot certify overlap. Chosen measured shortlist plus bounded full comparison preserves that evidence boundary.
