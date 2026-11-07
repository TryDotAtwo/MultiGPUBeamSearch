# Working capacity validation plan

Preserve native architecture, inference-first selection and exact fixed-width behavior. For beam_width=max, require full-frontier calibration regardless of the optional fixed-width flag; refuse an unverified result. Capacity discovery remains allocation-only evidence.

After full verification, compare seconds/parent against an available valid smaller full-frontier cache receipt on the same graph/models/GPU cohort/backend/build/precision/inference batch. Report allocation bound, verified width, reference width, throughput ratio and an explicitly bounded efficient recommendation; no claim of a universal efficient maximum. Missing comparable reference remains unverified. Do not reduce an explicit numeric beam.

Avoid carrying the allocation-search fixed8192 flush into production: remove only that search-only override when handing the admitted geometry to the runtime native planner, then fresh-admit; native memory authority remains unchanged.

Tests: max options force full, reject component-only, valid reference versus mismatch/missing/slowcase. GPU first measure smallreference then max on one faster12GB GPU if remaining budget/time admits it. First package used40.2min/~$0.072; remainder<80min/$1.928. New lease ceiling75min, no additional budget. Preserve logs and delete sole instance. If full boundary cannot finish within remaining time, report the exact gap.
