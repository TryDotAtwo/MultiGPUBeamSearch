#!/usr/bin/env python3
"""Compute a bounded phase timeout from the campaign's absolute billing deadline."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_FLOOR
import sys


def _decimal(value, name):
    try:
        number = Decimal(str(value))
    except (InvalidOperation, TypeError):
        raise ValueError(f"{name} must be a finite decimal number") from None
    if not number.is_finite():
        raise ValueError(f"{name} must be a finite decimal number")
    return number


def _whole_seconds(value, name, *, allow_zero):
    number = _decimal(value, name)
    minimum = 0 if allow_zero else 1
    if number != number.to_integral_value() or number < minimum:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a {qualifier} whole number of seconds")
    return int(number)


def phase_timeout_seconds(
    deadline_epoch,
    *,
    stop_reserve_seconds,
    future_reserve_seconds,
    phase_overhead_seconds=0,
    termination_grace_seconds=0,
    max_seconds,
    min_seconds,
    phase,
    now_epoch=None,
):
    deadline = _decimal(deadline_epoch, "deadline_epoch")
    now = _decimal(
        now_epoch if now_epoch is not None else datetime.now(timezone.utc).timestamp(),
        "now_epoch",
    )
    if deadline <= 0 or now <= 0:
        raise ValueError("deadline_epoch and now_epoch must be positive")
    stop_reserve = _whole_seconds(stop_reserve_seconds, "stop_reserve_seconds", allow_zero=True)
    future_reserve = _whole_seconds(future_reserve_seconds, "future_reserve_seconds", allow_zero=True)
    phase_overhead = _whole_seconds(phase_overhead_seconds, "phase_overhead_seconds", allow_zero=True)
    termination_grace = _whole_seconds(termination_grace_seconds, "termination_grace_seconds", allow_zero=True)
    maximum = _whole_seconds(max_seconds, "max_seconds", allow_zero=False)
    minimum = _whole_seconds(min_seconds, "min_seconds", allow_zero=False)
    if minimum > maximum:
        raise ValueError("min_seconds cannot exceed max_seconds")
    remaining = int((deadline - now).to_integral_value(rounding=ROUND_FLOOR))
    available = remaining - stop_reserve - future_reserve - phase_overhead - termination_grace
    if available < minimum:
        raise ValueError(
            f"{phase} cannot start: {max(0, available)} command seconds remain after stop, future, "
            f"phase-overhead and termination-grace reserves, but at least {minimum} are required"
        )
    return min(maximum, available)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deadline-epoch", required=True)
    parser.add_argument("--stop-reserve-seconds", required=True, type=int)
    parser.add_argument("--future-reserve-seconds", required=True, type=int)
    parser.add_argument("--phase-overhead-seconds", type=int, default=0)
    parser.add_argument("--termination-grace-seconds", type=int, default=0)
    parser.add_argument("--max-seconds", required=True, type=int)
    parser.add_argument("--min-seconds", required=True, type=int)
    parser.add_argument("--phase", required=True)
    args = parser.parse_args(argv)
    try:
        seconds = phase_timeout_seconds(
            args.deadline_epoch,
            stop_reserve_seconds=args.stop_reserve_seconds,
            future_reserve_seconds=args.future_reserve_seconds,
            phase_overhead_seconds=args.phase_overhead_seconds,
            termination_grace_seconds=args.termination_grace_seconds,
            max_seconds=args.max_seconds,
            min_seconds=args.min_seconds,
            phase=args.phase,
        )
    except ValueError as error:
        parser.error(str(error))
    print(seconds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
