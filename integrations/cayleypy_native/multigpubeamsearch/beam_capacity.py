"""Bounded capacity search using native admission, never a memory estimate.

Admission is monotone only for a frozen profile and memory snapshot. Search
each supplied profile independently. Runtime, timeout and transport failures
must propagate; only CapacityRejected means an actual memory/capacity bound.
"""
from __future__ import annotations
from dataclasses import dataclass
import time


class CapacityRejected(ValueError):
    """The native planner explicitly rejected this beam's capacity."""


@dataclass(frozen=True)
class CapacityResult:
    requested_beam: int
    effective_beam: int
    profile: dict
    plans: tuple
    probes: int
    upper_rejected_beam: int | None
    search_complete: bool
    full_step_verified: bool = False


def find_capacity(profiles, *, admit, upper_bound, alignment, deadline,
                  verify_full_step=None):
    """Largest admitted aligned beam among fixed profiles and a given ceiling.

    admit(beam, profile) must return all-rank validated native plans or raise
    CapacityRejected. No subprocess failure can stand in for an OOM receipt.
    An unfinished search returns its safe lower bound with search_complete=False.
    Full-step verification requires an explicit callback and the same plans.
    """
    from .pipeline_profiles import validate_rank_plans
    if type(upper_bound) is not int or type(alignment) is not int:
        raise ValueError('integer beam ceiling and alignment required')
    if alignment < 1 or upper_bound < alignment:
        raise ValueError('beam ceiling must accommodate one aligned frontier')
    if not callable(admit):
        raise TypeError('native admission callback required')
    best = None
    probes = 0
    complete = True
    maximum = upper_bound // alignment
    for supplied in profiles:
        profile = dict(supplied)
        low, high = 0, maximum + 1
        accepted = None
        rejected = None
        # Binary search is confined to a FIXED profile; an automatic planner
        # choosing different layouts at each query cannot provide monotonicity.
        while high - low > 1:
            if time.monotonic() >= deadline:
                complete = False
                break
            units = (low + high) // 2
            beam = units * alignment
            probes += 1
            try:
                plans = tuple(admit(beam, dict(profile)))
                validate_rank_plans(plans)
                effective = plans[0]['GLOBAL_BEAM_WIDTH_EFFECTIVE']
                if effective != beam:
                    raise ValueError('capacity search requires exact frozen alignment')
            except CapacityRejected:
                high, rejected = units, beam
            else:
                low, accepted = units, plans
        if accepted is not None and (best is None or low * alignment > best.requested_beam):
            best = CapacityResult(low * alignment, low * alignment, profile,
                                  accepted, probes, rejected, complete)
        if not complete:
            break
    if best is None:
        raise CapacityRejected('no frontier admitted within the search budget')
    verified = False
    if verify_full_step is not None:
        if not callable(verify_full_step):
            raise TypeError('full-step verifier must be callable')
        verified = verify_full_step(best.requested_beam, dict(best.profile), best.plans) is True
        if not verified:
            raise ValueError('largest admitted frontier failed full-step verification')
    return CapacityResult(best.requested_beam, best.effective_beam, best.profile,
                          best.plans, probes, best.upper_rejected_beam, complete, verified)
