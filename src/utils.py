"""Small utilities shared by the executable pipeline."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator


@contextmanager
def timer() -> Iterator[dict[str, float]]:
    """Measure a block and expose elapsed seconds through a mutable result."""
    result = {"elapsed": 0.0}
    start = time.perf_counter()
    try:
        yield result
    finally:
        result["elapsed"] = time.perf_counter() - start

