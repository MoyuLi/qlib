"""AI layer.

Turns unstructured market data (news, filings) into numeric, point-in-time
signals the strategy layer can blend with classical factors:

* ``sentiment``  — FinBERT / FinGPT scoring of headlines (transformers).
* ``events``     — LLM event extraction into structured ``ExtractedEvent``s.
* ``signals``    — aggregate the above into a per-symbol cross-sectional score.

Every model/SDK import is lazy and degrades gracefully: if transformers / a key
is missing, the scorer falls back to a fast lexicon model so the pipeline still
produces a (weaker) signal rather than crashing.
"""
