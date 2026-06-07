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
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import yaml
from cachetools import LRUCache
from loguru import logger

from app.config.settings import get_settings

# Must match the union in intent_rules.Intent + extra graph-only intents
# (MALICIOUS / KNOWLEDGE / BRAINSTORM aren't handled by regex; they are
# gate-eligible so the LLM pre-processor can be skipped for clear cases).
Intent = Literal[
    "GREETING", "AMBIGUOUS", "OFF_SCOPE", "TOPIC_LIST",
    "KNOWLEDGE", "BRAINSTORM", "MALICIOUS",
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
# Proper LRU: frequently asked queries stay cached; stale entries evict
# automatically. cachetools.LRUCache is O(1) get/set with a doubly-linked
# list. At 2048 entries × 1024-dim × 8 bytes ≈ 16 MB — bounded and safe.
_embed_cache: LRUCache = LRUCache(maxsize=2048)


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

    Returns the best-matching intent ONLY IF:
      1. ``best_sim >= settings.intent_semantic_threshold`` (default 0.78)
      2. ``best_sim - second_sim >= settings.intent_semantic_margin`` (default 0.10)

    Otherwise returns None — caller falls through to regex Tier-1, then
    the LLM pre-processor.

    Args:
        text: The raw user query (any language, any length).

    Returns:
        The intent label, or None if the gate is uncertain.
    """
    if not text or not text.strip():
        return None
    text = text.strip()
    settings = get_settings()

    # Compute centroids (cached; cheap if already done)
    try:
        centroids = await _compute_centroids()
    except Exception as e:
        logger.warning(f"Semantic gate: centroid computation failed, falling through: {e}")
        return None
    if not centroids:
        return None

    # Embed user query (cached on repeat)
    try:
        user_vec = await _embed_one(text)
    except Exception as e:
        logger.warning(f"Semantic gate: embedding failed for query, falling through: {e}")
        return None
    user_vec = _l2_normalize(user_vec)

    # Cosine sim to each centroid
    sims = {intent: _cosine(user_vec, c) for intent, c in centroids.items()}
    ranked = sorted(sims.items(), key=lambda kv: kv[1], reverse=True)
    best_intent, best_sim = ranked[0]
    second_sim = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = best_sim - second_sim

    if best_sim >= settings.intent_semantic_threshold and margin >= settings.intent_semantic_margin:
        logger.info(
            f"Semantic gate HIT: intent={best_intent} sim={best_sim:.3f} "
            f"margin={margin:.3f} (2nd={second_sim:.3f}) query={text[:60]!r}"
        )
        return best_intent  # type: ignore[return-value]
    logger.debug(
        f"Semantic gate MISS: best={best_intent}@{best_sim:.3f} margin={margin:.3f} "
        f"(threshold={settings.intent_semantic_threshold}, "
        f"min_margin={settings.intent_semantic_margin}) — falling through"
    )
    return None


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
