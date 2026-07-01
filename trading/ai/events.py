"""LLM event extraction.

Distills free-text (a news headline+summary, or a filing excerpt) into a
structured ``ExtractedEvent`` — event type, directional polarity in [-1, 1],
and a confidence — so the strategy layer can react to *what happened*, not just
an undifferentiated sentiment blob.

Backends:

* ``anthropic`` — Claude via the official SDK (needs ``ANTHROPIC_API_KEY``).
* ``openai``    — GPT via the official SDK (needs ``OPENAI_API_KEY``).
* ``rules``     — keyword classifier, no API. Automatic fallback.

The prompt asks for strict JSON and we parse defensively; any failure falls back
to the rules backend so the pipeline never hard-crashes on a model hiccup.
"""
from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime
from typing import List, Optional, Sequence

from trading.data.sources.base import ExtractedEvent, NewsItem

log = logging.getLogger(__name__)

EVENT_TYPES = [
    "earnings", "guidance", "mna", "regulatory", "legal", "product",
    "management", "macro", "analyst_rating", "other",
]

_SYSTEM = (
    "You are a sell-side analyst. Classify the financial news item into exactly "
    "one event_type from this list: " + ", ".join(EVENT_TYPES) + ". Estimate its "
    "directional impact on the issuer's equity as 'polarity' in [-1,1] (negative "
    "= bearish) and your 'confidence' in [0,1]. Respond with ONLY a JSON object: "
    '{"event_type": str, "polarity": float, "confidence": float, "evidence": str}.'
)

_RULES = {
    "earnings": ["earnings", "eps", "revenue", "quarter", "beat", "miss"],
    "guidance": ["guidance", "outlook", "forecast", "raises", "cuts forecast"],
    "mna": ["acqui", "merger", "buyout", "takeover", "stake"],
    "regulatory": ["fda", "approval", "sec", "antitrust", "regulat"],
    "legal": ["lawsuit", "settlement", "fraud", "probe", "investigation"],
    "product": ["launch", "unveil", "recall", "release"],
    "management": ["ceo", "cfo", "resign", "appoint", "executive"],
    "analyst_rating": ["upgrade", "downgrade", "price target", "initiat"],
}


class EventExtractor:
    def __init__(self, backend: str = "auto", model: Optional[str] = None):
        # "auto": use whichever LLM key is present (Anthropic preferred), else rules.
        if backend == "auto":
            if os.environ.get("ANTHROPIC_API_KEY"):
                backend = "anthropic"
            elif os.environ.get("OPENAI_API_KEY"):
                backend = "openai"
            else:
                backend = "rules"
        self.backend = backend
        self.model = model
        self._client = None

    def extract(self, items: Sequence[NewsItem]) -> List[ExtractedEvent]:
        out: List[ExtractedEvent] = []
        for item in items:
            text = f"{item.headline}. {item.summary}".strip()
            parsed = None
            if self.backend == "anthropic":
                parsed = self._via_anthropic(text)
            elif self.backend == "openai":
                parsed = self._via_openai(text)
            if parsed is None:
                parsed = self._via_rules(text)
            out.append(
                ExtractedEvent(
                    symbol=item.symbol,
                    datetime=item.datetime or datetime.utcnow(),
                    event_type=parsed.get("event_type", "other"),
                    polarity=_clamp(parsed.get("polarity", 0.0), -1, 1),
                    confidence=_clamp(parsed.get("confidence", 0.0), 0, 1),
                    evidence=str(parsed.get("evidence", ""))[:300],
                )
            )
        return out

    # --- Anthropic -----------------------------------------------------------
    def _via_anthropic(self, text: str) -> Optional[dict]:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return None
        try:
            if self._client is None:
                import anthropic

                self._client = anthropic.Anthropic()
            resp = self._client.messages.create(
                model=self.model or "claude-haiku-4-5-20251001",
                max_tokens=256,
                system=_SYSTEM,
                messages=[{"role": "user", "content": text}],
            )
            return _parse_json("".join(b.text for b in resp.content if b.type == "text"))
        except Exception as e:
            log.warning("anthropic event extraction failed: %s", e)
            return None

    # --- OpenAI --------------------------------------------------------------
    def _via_openai(self, text: str) -> Optional[dict]:
        if not os.environ.get("OPENAI_API_KEY"):
            return None
        try:
            if self._client is None:
                from openai import OpenAI

                self._client = OpenAI()
            resp = self._client.chat.completions.create(
                model=self.model or "gpt-4o-mini",
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": text},
                ],
            )
            return _parse_json(resp.choices[0].message.content)
        except Exception as e:
            log.warning("openai event extraction failed: %s", e)
            return None

    # --- rules fallback ------------------------------------------------------
    @staticmethod
    def _via_rules(text: str) -> dict:
        low = text.lower()
        event_type = "other"
        for etype, kws in _RULES.items():
            if any(kw in low for kw in kws):
                event_type = etype
                break
        # polarity from VADER (no LLM, no torch)
        from trading.ai.sentiment import SentimentScorer

        polarity = SentimentScorer(backend="vader").score([text])[0]
        return {
            "event_type": event_type,
            "polarity": polarity,
            "confidence": 0.3 if event_type != "other" else 0.1,
            "evidence": text[:200],
        }


def _parse_json(s: str) -> Optional[dict]:
    if not s:
        return None
    match = re.search(r"\{.*\}", s, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _clamp(x, lo, hi) -> float:
    try:
        return max(lo, min(hi, float(x)))
    except (TypeError, ValueError):
        return 0.0
