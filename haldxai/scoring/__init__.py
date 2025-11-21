# -*- coding: utf-8 -*-
"""
Scoring modules for HALDxAI platform.
"""

from .aging_relevance import AgingScorer
from .bridge_candidates import BridgeCandidateScorer

__all__ = [
    "AgingScorer",
    "BridgeCandidateScorer",
]