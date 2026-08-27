# -*- coding: utf-8 -*-
"""Shared concurrency signal for external (non-benchmark) requests.

The benchmark scheduler tags its requests with ``x-locallm-benchmark``.  The
gateway counts requests that do *not* carry that header (real user traffic) in
a thread-safe counter.  The scheduler reads :func:`external_active` and backs off
before sending each item, so external requests are served ahead of benchmark
work — downgrade-priority preemption (the benchmark never pauses; it just
yields the backend when real traffic is present).
"""

from __future__ import annotations

import threading

_lock = threading.Lock()
_count = 0


def external_started() -> None:
    global _count
    with _lock:
        _count += 1


def external_finished() -> None:
    global _count
    with _lock:
        if _count > 0:
            _count -= 1


def external_active() -> int:
    with _lock:
        return _count
