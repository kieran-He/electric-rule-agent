from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class VerificationResult:
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    confidence: float = 0.0
    needs_retry: bool = False
    warning: Optional[str] = None
    verification_type: str = "none"