"""Semantic evidence retrieval boundary with a model-missing fallback.

Phase 6 of the roadmap: a semantic layer is only enabled when the fixed
evaluation set shows clear rule/classifier misses. This module defines the
port (in ports.py) and the disabled-by-default adapter. The adapter never
blocks the rules/classifier/state-machine chain: with no model it reports
`available=False` and returns no evidence.
"""

from __future__ import annotations

import logging
from typing import Any

from app.modules.fraud.ports import SemanticEvidenceRetriever

logger = logging.getLogger(__name__)


class DisabledSemanticEvidenceRetriever:
    """No-op adapter used while semantic retrieval is disabled or not installed."""

    def __init__(self, *, enabled: bool = False) -> None:
        self._enabled = enabled

    @property
    def available(self) -> bool:
        return False

    async def retrieve(self, *, text: str, session_id: str) -> list[dict[str, Any]]:
        return []


def build_semantic_retriever(*, enabled: bool) -> SemanticEvidenceRetriever:
    """Return the configured retriever; always usable, never required.

    The embedding-backed adapter is intentionally not implemented in this
    phase: it must be introduced only after the fixed eval set demonstrates
    semantic misses, and its latency must be accounted into PARTIAL/FINAL.
    """
    if not enabled:
        return DisabledSemanticEvidenceRetriever(enabled=False)
    return DisabledSemanticEvidenceRetriever(enabled=True)
