"""Re-export shim — FailureMode now lives in safety_probe.categories.failures."""

from __future__ import annotations

from safety_probe.categories.failures import FailureMode

__all__ = ["FailureMode"]
