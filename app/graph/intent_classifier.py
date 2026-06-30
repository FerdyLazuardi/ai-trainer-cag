"""
Tier-0 semantic intent classifier — embedding-based fast path.

Replaces the regex-only Tier-1 with a cosine-similarity gate against
pre-computed intent centroids. This generalises across languages and
cultural contexts WITHOUT hand-maintained regex patterns.

Why this exists
---------------
The regex Tier-1 in ``intent_rules.py`` is fast and free, but only matches
patterns its author has seen. A 13k-user fleet produces greeting variants
the author has *not* seen (religious greetings, regional slang, elongation
typos, novel identity questions). Anything Tier-1 misses falls through to
the LLM pre-processor, which costs ~2.7K input tokens and ~5 s of latency
to produce an intent label that could have been a cosine sim.

This module sits between Tier-1 (regex) and the LLM pre-processor. It is
called only when Tier-1 returns None, so its ~200 ms embedding cost is
paid only on the long tail of novel inputs.

Design properties
-----------------
* NO hardcoded language patterns. The seed file (``intent_seed.yaml``)
  provides semantic anchors; the embedding model generalises from them.
  Adding a new example to the YAML automatically extends coverage.
* Decision rule: best-centroid similarity must clear
  ``settings.intent_semantic_threshold`` AND beat the runner-up by
  ``settings.intent_semantic_margin``. Otherwise returns None and the
  caller falls through to the LLM.
* Centroids are pre-computed (mean of seed example embeddings, L2-normalised
  so dot product = cosine) and cached for the process lifetime. Cache
  invalidates automatically when the seed file fingerprint changes.
* Bootstrap-friendly: the seed file is intentionally small and curated.
  Once production traffic exists, recompute centroids from intent-labelled
  agent_logs queries (script stub: ``scripts/recompute_intent_centroids.py``).

Cost
----
~200 ms for the embedding call (cached on repeat) + ~1 ms cosine. Vs the
LLM pre-processor at ~2.7K tokens / ~5 s, this is ~25x cheaper and ~25x
faster for any query the gate catches. Failures (low confidence) cost
only the ~200 ms of the embedding call before falling through.
"""
from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import yaml
from collections import OrderedDict
from loguru import logger


class _LRU(OrderedDict):
    """Minimal bounded LRU cache (stdlib only — no cachetools dependency).

    cachetools was NOT installed in the runtime container and not declared in
    requirements.txt; importing it here used to make this whole module
    un-importable at runtime (it only worked because the module was dead code).
    This OrderedDict-based LRU is O(1) get/set, evicts the oldest entry past
    maxsize, and keeps the same .get()/[]=  interface the call sites use.
    """
    def __init__(self, maxsize: int = 2048):
        super().__init__()
        self._maxsize = maxsize

    def get(self, key, default=None):
        if key not in self:
            return default
        self.move_to_end(key)
        return self[key]

    def __setitem__(self, key, value):
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        while len(self) > self._maxsize:
            super().__delitem__(next(iter(self)))

from app.config.settings import get_settings

# Gate-eligible intents the semantic classifier can commit. Mirrors the
# no-retrieval buckets in _pre_processor (regex Tier-1) — never KNOWLEDGE/
# COACHING (those need retrieval) and never MALICIOUS (stays on the regex guard).
Intent = Literal[
    "GREETING", "AMBIGUOUS", "OFF_SCOPE", "TOPIC_LIST",
]


# ── Seed loading ──────────────────────────────────────────────────────────────
_seed_cache: dict[str, list[str]] | None = None
_seed_path: Path = Path(__file__).parent / "intent_seed.yaml"


def _load_seed_examples() -> dict[str, list[str]]:
    """Load seed examples from the YAML file. Cached for the process lifetime.

    Returns an empty dict if the file is missing or malformed — the gate then
    becomes a no-op (returns None for everything) and the existing Tier-1
    regex + LLM pre-processor handle the traffic unchanged.
    """
    global _seed_cache
    if _seed_cache is not None:
        return _seed_cache
    if not _seed_path.exists():
        logger.warning(
            f"Intent seed file not found at {_seed_path} — semantic gate disabled "
            f"(classifier will return None for all queries, falling through to "
            f"regex Tier-1 then LLM pre-processor)"
        )
        _seed_cache = {}
        return _seed_cache
    try:
        with open(_seed_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        # Defensive: only keep string-list entries
        cleaned = {k: v for k, v in data.items() if isinstance(v, list) and all(isinstance(x, str) for x in v)}
        _seed_cache = cleaned
        total = sum(len(v) for v in cleaned.values())
        logger.info(f"Loaded {len(cleaned)} intents / {total} seed examples from {_seed_path.name}")
        return _seed_cache
    except Exception as e:
        logger.error(f"Failed to load intent seed ({_seed_path}): {e} — semantic gate disabled")
        _seed_cache = {}
        return _seed_cache


def invalidate_seed_cache() -> None:
    """Clear all in-memory caches. Call after editing intent_seed.yaml or
    after running a recompute script. Also useful in tests."""
    global _seed_cache
    _seed_cache = None
    _centroid_cache.clear()
    _centroid_fingerprint = ""
    _embed_cache.clear()


# ── Embedding cache ───────────────────────────────────────────────────────────
# Bounded LRU (stdlib OrderedDict, see _LRU above): frequently asked queries
# stay cached; oldest entries evict past maxsize. At 2048 entries × 1024-dim ×
# 8 bytes ≈ 16 MB — bounded and safe.
_embed_cache: _LRU = _LRU(maxsize=2048)


async def _embed_one(text: str) -> list[float]:
    """Embed a single string via the configured embed model, with LRU cache."""
    cached = _embed_cache.get(text)
    if cached is not None:
        return cached
    from llama_index.core import Settings
    # Defensive: ensure the embed model is initialized. The retrieval path
    # calls ensure_llamaindex_configured() in its own startup; the gate may
    # be called before that, or in isolation (tests, warmup). Calling it
    # here is idempotent and cheap after the first call.
    from app.config.embedding_config import ensure_llamaindex_configured
    try:
        ensure_llamaindex_configured()
    except Exception:
        pass  # Let the actual embed call raise; classifier fallback handles it.
    vec = await Settings.embed_model.aget_query_embedding(text)
    _embed_cache[text] = vec
    return vec


# ── Centroid computation ─────────────────────────────────────────────────────
_centroid_cache: dict[str, list[float]] = {}
_centroid_fingerprint: str = ""


def _seed_fingerprint(seed: dict[str, list[str]]) -> str:
    """Content fingerprint so cache auto-invalidates when the seed file changes."""
    h = hashlib.sha256()
    for intent in sorted(seed.keys()):
        h.update(intent.encode("utf-8"))
        for ex in seed[intent]:
            h.update(b"\x00")
            h.update(ex.encode("utf-8"))
    return h.hexdigest()


def _l2_normalize(v: list[float]) -> list[float]:
    norm = sum(x * x for x in v) ** 0.5
    if norm == 0:
        return v
    return [x / norm for x in v]


async def _compute_centroids() -> dict[str, list[float]]:
    """Compute mean embedding per intent class. L2-normalised so dot product
    is equivalent to cosine similarity at compare time.

    Cached for the process lifetime. Cache invalidates when the seed file
    content changes (detected via SHA-256 fingerprint of the YAML payload).
    """
    global _centroid_fingerprint
    seed = _load_seed_examples()
    fp = _seed_fingerprint(seed)
    if fp == _centroid_fingerprint and _centroid_cache:
        return _centroid_cache

    centroids: dict[str, list[float]] = {}
    
    # 1. Gather all unique examples across all intents
    _examples_set: set[str] = set()
    for examples in seed.values():
        _examples_set.update(examples)
    all_examples: list[str] = list(_examples_set)
    
    if not all_examples:
        return centroids
        
    # 2. Check cache for hits; collect misses
    misses = [ex for ex in all_examples if _embed_cache.get(ex) is None]
    
    # 3. Batch embed the misses (single round-trip to the embedding API)
    if misses:
        from llama_index.core import Settings
        from app.config.embedding_config import ensure_llamaindex_configured
        try:
            ensure_llamaindex_configured()
        except Exception:
            pass
        batch_vectors = await Settings.embed_model.aget_text_embedding_batch(misses)
        for i, ex in enumerate(misses):
            _embed_cache[ex] = batch_vectors[i]
            
    # 4. Compute centroids per intent using the now-cached vectors
    for intent, examples in seed.items():
        if not examples:
            continue
        vectors: list[list[float]] = [v for ex in examples if (v := _embed_cache.get(ex)) is not None]
        dim = len(vectors[0])
        mean = [0.0] * dim
        for vec in vectors:
            for i, v in enumerate(vec):
                mean[i] += v
        mean = [v / len(vectors) for v in mean]
        centroids[intent] = _l2_normalize(mean)

    _centroid_fingerprint = fp
    _centroid_cache.clear()
    _centroid_cache.update(centroids)
    logger.info(
        f"Computed {len(centroids)} intent centroids "
        f"({sum(len(v) for v in seed.values())} seed examples total)"
    )
    return centroids


# ── Public API ────────────────────────────────────────────────────────────────
def _cosine(a: list[float], b: list[float]) -> float:
    """Dot product via numpy — releases the GIL during the C-level computation.

    Both inputs are assumed L2-normalised, so dot product = cosine similarity.
    On 1024-dim vectors this is ~100x faster than the pure-Python zip+sum and,
    critically, does NOT hold the GIL during the multiply-accumulate — so the
    asyncio event loop stays responsive under concurrent requests on 2 vCPUs.
    """
    return float(np.dot(a, b))


async def classify_semantic(text: str) -> Optional[Intent]:
    """Classify intent via embedding similarity to pre-computed centroids.

    Thin wrapper kept for back-compat — returns only the committed intent
    (or None). For per-decision telemetry (HIT/MISS + best/second cosine +
    margin), call :func:`classify_semantic_with_scores` and inspect
    ``result.decision`` + ``result.best_*``/``result.margin``.
    """
    res = await classify_semantic_with_scores(text)
    return res.committed


@dataclass
class GateScore:
    """Per-decision telemetry from the semantic intent gate.

    Returned by :func:`classify_semantic_with_scores`. The chat route writes
    these to ``agent_logs.gate_*`` so the calibration harness and the
    Streamlit dashboard can plot distributions and detect drift.

    Fields
    ------
    decision: "HIT" if the gate committed an intent, "MISS" if it ran but
      stayed silent (caller fell through), "SKIP" if the gate didn't run
      at all (e.g. caller already had a regex match).
    committed: the intent the gate committed to, or None on MISS/SKIP.
    best_intent / best_cosine: the winning centroid + its cosine.
    second_intent / second_cosine: runner-up; second_cosine defaults to
      0.0 when there is no runner-up (single-centroid seeds).
    margin: best_cosine - second_cosine.
    """
    decision: Literal["HIT", "MISS", "SKIP"]
    committed: Optional[Intent]
    best_intent: Optional[str]
    best_cosine: float
    second_intent: Optional[str]
    second_cosine: float
    margin: float


async def classify_semantic_with_scores(
    text: str,
    query_embedding: list[float] | None = None,
) -> GateScore:
    """Classify intent AND return the full score breakdown.

    Returns a :class:`GateScore`. The caller decides what to do with
    ``decision`` — `_pre_processor` commits the intent on HIT; the chat
    route writes all fields to ``agent_logs`` regardless of HIT/MISS so the
    drift monitor can see *near-misses* too.
    """
    empty = GateScore(
        decision="SKIP", committed=None,
        best_intent=None, best_cosine=0.0,
        second_intent=None, second_cosine=0.0,
        margin=0.0,
    )
    if not text or not text.strip():
        return empty
    text = text.strip()
    settings = get_settings()

    try:
        centroids = await _compute_centroids()
    except Exception as e:
        logger.warning(f"Semantic gate: centroid computation failed, falling through: {e}")
        return empty
    if not centroids:
        return empty

    try:
        if query_embedding is not None:
            user_vec = list(query_embedding)
        else:
            user_vec = await _embed_one(text)
    except Exception as e:
        logger.warning(f"Semantic gate: embedding failed for query, falling through: {e}")
        return empty
    user_vec = _l2_normalize(user_vec)

    sims = {intent: _cosine(user_vec, c) for intent, c in centroids.items()}
    ranked = sorted(sims.items(), key=lambda kv: kv[1], reverse=True)
    best_intent, best_sim = ranked[0]
    second_intent, second_sim = (ranked[1][0], ranked[1][1]) if len(ranked) > 1 else (None, 0.0)
    margin = best_sim - second_sim

    # Defensive: the local Intent union is the 4 gate-eligible ones, but the
    # centroids dict may carry KNOWLEDGE/COACHING/MALICIOUS from seed — those
    # must never win via this gate. Filter to gate-eligible only.
    _GATE_OK = {"GREETING", "AMBIGUOUS", "OFF_SCOPE", "TOPIC_LIST"}
    if best_intent not in _GATE_OK:
        return GateScore(
            decision="MISS", committed=None,
            best_intent=best_intent, best_cosine=best_sim,
            second_intent=second_intent, second_cosine=second_sim,
            margin=margin,
        )

    if best_sim >= settings.intent_semantic_threshold and margin >= settings.intent_semantic_margin:
        logger.info(
            f"Semantic gate HIT: intent={best_intent} sim={best_sim:.3f} "
            f"margin={margin:.3f} (2nd={second_sim:.3f}) query={text[:60]!r}"
        )
        return GateScore(
            decision="HIT", committed=best_intent,  # type: ignore[arg-type]
            best_intent=best_intent, best_cosine=best_sim,
            second_intent=second_intent, second_cosine=second_sim,
            margin=margin,
        )
    logger.debug(
        f"Semantic gate MISS: best={best_intent}@{best_sim:.3f} margin={margin:.3f} "
        f"(threshold={settings.intent_semantic_threshold}, "
        f"min_margin={settings.intent_semantic_margin}) — falling through"
    )
    return GateScore(
        decision="MISS", committed=None,
        best_intent=best_intent, best_cosine=best_sim,
        second_intent=second_intent, second_cosine=second_sim,
        margin=margin,
    )


# ── Coaching affinity (auto-hook) ─────────────────────────────────────────────
async def coaching_affinity(text: str, query_embedding: Optional[list[float]] = None) -> float:
    """Cosine similarity of `text` to the COACHING centroid, in [0, 1]-ish.

    Used by the chat route's auto-hook to decide whether to OFFER coaching after
    a normal answer (it never auto-activates). NOT a gate — just a soft score the
    frontend turns into a one-line offer above settings.coaching_suggest_threshold.

    Reuses a precomputed `query_embedding` when given (the route already embeds
    the query for LTM), so this adds ~1ms (one dot product), no extra embed call.
    Returns 0.0 on any failure or if the COACHING centroid is unavailable.
    """
    if not text or not text.strip():
        return 0.0
    try:
        centroids = await _compute_centroids()
        c = centroids.get("COACHING")
        if not c:
            return 0.0
        if query_embedding is not None:
            vec = query_embedding
        else:
            vec = await _embed_one(text.strip())
        return _cosine(_l2_normalize(vec), c)
    except Exception as e:
        logger.debug(f"coaching_affinity failed (non-fatal): {e}")
        return 0.0


# ── Semantic TOPIC_LIST fallback ──────────────────────────────────────────────
async def is_topic_list_semantic(
    text: str, threshold: float = 0.70, query_embedding: Optional[list[float]] = None
) -> bool:
    """True if `text` semantically reads as a 'what topics/materials can I learn?'
    question — used as a FALLBACK after the regex Tier-1 misses (typos, paraphrases
    like "materi yang kamu bisa pelajari", "bisa belajar apa hari ini").

    Strict gate: the COACHING/KNOWLEDGE centroids sit near TOPIC_LIST, so we
    require BOTH (a) TOPIC_LIST is the single best-matching centroid AND
    (b) its cosine >= threshold. Calibrated 2026-06-10: real topic-list phrasings
    score 0.74-0.86 with TOPIC_LIST as best; "produk amartha apa aja" (a content
    question that must stay KNOWLEDGE) scores 0.58 with KNOWLEDGE as best — clean
    separation. Reuses a precomputed `query_embedding` (route already embeds the
    query for LTM/cache) when given, so this adds ~1ms on the hot path. Returns
    False on any failure (degrade to regex/KNOWLEDGE).
    """
    if not text or not text.strip():
        return False
    try:
        centroids = await _compute_centroids()
        if "TOPIC_LIST" not in centroids:
            return False
        vec = query_embedding if query_embedding is not None else await _embed_one(text.strip())
        vec = _l2_normalize(vec)
        sims = {k: _cosine(vec, c) for k, c in centroids.items()}
        best = max(sims, key=lambda k: sims[k])
        return best == "TOPIC_LIST" and sims["TOPIC_LIST"] >= threshold
    except Exception as e:
        logger.debug(f"is_topic_list_semantic failed (non-fatal): {e}")
        return False


# ── Warmup helper (optional) ──────────────────────────────────────────────────
async def warmup() -> None:
    """Pre-compute centroids at startup so the first user request doesn't
    pay the seed-embedding cost. Safe to call multiple times; subsequent
    calls hit the centroid cache.

    Wire into app/main.py lifespan if you want true cold-start warmup. The
    first user request is still fast even without this (~200 ms extra) but
    this eliminates the spike.
    """
    try:
        await _compute_centroids()
    except Exception as e:
        logger.warning(f"Intent gate warmup failed (non-fatal): {e}")
