# Askfer — Portfolio Chat Assistant — Design Spec

**Date:** 2026-05-23
**Author:** Ferdy Fadhil Lazuardi (with Claude)
**Status:** Approved, ready for implementation plan

---

## 1. Goal

Add a new public-facing chat endpoint **Askfer** that answers questions about Ferdy's portfolio (10 projects + homepage + CV) on `https://ferdy-fadhil-lazuardi.my.id/`. Askfer is a parallel pipeline to the existing **A-Pedi** (Amarthapedia) assistant — it must NOT modify A-Pedi's runtime behavior.

### Success criteria

1. `POST /api/v1/askfer/stream` returns an SSE stream answering project/CV questions in first-person as Ferdy.
2. `POST /api/v1/askfer/sync` (admin-secret protected) ingests homepage + 10 projects + CV PDF into a new Qdrant collection `Personal_Portfolio`.
3. Existing `/api/v1/chat` and `/api/v1/chat/stream` continue to work unchanged, still hitting the `Knowledge_Base` collection.
4. Rate limit by IP: 10 req/min/IP for `/askfer/stream`; the 11th request within a minute gets HTTP 429.
5. Off-scope query → polite redirect; not-found query → honest "tanya via LinkedIn/email".
6. Bilingual auto-detect: English query → English response; Indonesian query → Indonesian response.
7. Source citations dedupe per project URL (one chunk-set per project surfaces as one source).

---

## 2. Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│  ferdy-fadhil-lazuardi.my.id  (portfolio website, public)        │
│   └─ embed chat widget → POST https://<ai>/api/v1/askfer/stream  │
└──────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│  FastAPI app (existing)                                          │
│                                                                  │
│  ─ A-Pedi (UNTOUCHED) ─────────────────                          │
│   /api/v1/chat, /chat/stream  →  graph/pipeline.py               │
│       ↳ Qdrant collection: Knowledge_Base                        │
│                                                                  │
│  ─ Askfer (NEW) ──────────────────────                           │
│   /api/v1/askfer/stream       →  graph/askfer_pipeline.py        │
│       ↳ Qdrant collection: Personal_Portfolio                    │
│                                                                  │
│   /api/v1/askfer/sync (admin) →  ingestion/portfolio_sync.py     │
│       ↳ scrape sitemap.xml + CV PDF                              │
└──────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│  Shared infra (reused, no changes)                               │
│   - Qdrant (new collection added)                                │
│   - Redis (rate-limit by IP, semantic cache w/ collection scope) │
│   - PostgreSQL (Document/Chunk records — collection-tagged)      │
│   - LLM client (OpenRouter), embedder, reranker (Cohere)         │
└──────────────────────────────────────────────────────────────────┘
```

### Isolation guarantees

- **Zero edits** to `app/graph/pipeline.py`, `app/api/routes/chat.py`, `app/llm/prompts.py`.
- The Askfer route does NOT depend on `get_current_user` (A-Pedi's JWT auth) — it uses a new dependency `rate_limit_by_ip`.
- Cache key includes a namespace prefix; A-Pedi keeps default namespace `"rag"` (preserves byte-identical `rag:cache:...` Redis keys), Askfer uses `"askfer"`.

### Shared (read-only reuse)

- `hybrid_search(collection=...)` — already collection-aware via parameter.
- `rerank()`, `validate_followups()`, `render_followup_block()`.
- `set_cached_response()`, `get_cached_response()` — extended with optional `cache_namespace` param.
- `get_llm()`, `get_cheap_llm()`, embedding configuration, Qdrant client.

---

## 3. Qdrant Collection — `Personal_Portfolio`

### Settings additions (`app/config/settings.py`)

```python
qdrant_personal_collection: str = "Personal_Portfolio"
portfolio_sitemap_url: str = "https://ferdy-fadhil-lazuardi.my.id/sitemap.xml"
portfolio_homepage_url: str = "https://ferdy-fadhil-lazuardi.my.id/"
portfolio_project_url_pattern: str = r"^https://ferdy-fadhil-lazuardi\.my\.id/projects/[^/]+/?$"
portfolio_cv_url: str = "https://ferdy-fadhil-lazuardi.my.id/CV%20-%20Ferdy%20Fadhil%20Lazuardi.pdf"
askfer_admin_secret: str = Field(default="", alias="ASKFER_ADMIN_SECRET")
askfer_rate_limit_per_minute: int = 10
```

### Collection configuration

Mirrors `Knowledge_Base` (hybrid dense + sparse):

- `text-dense`: 1536-dim (matches `text-embedding-3-small`), cosine distance.
- `text-sparse`: BM25 modifier IDF.
- HNSW: `m=16`, `ef_construct=100`.
- `on_disk_payload=True`.
- Payload indexes (KEYWORD): `document_id`, `source`, `doc_type`.

### `doc_type` discriminator (KEYWORD index)

- `"homepage"` — chunks from `https://ferdy-fadhil-lazuardi.my.id/`.
- `"project"` — chunks from `/projects/<slug>/` pages (10 entries expected).
- `"cv"` — chunks from CV PDF.

This field enables future intent-biased retrieval (e.g., skill questions boost `cv`, project questions boost `project`). For MVP, retrieval treats all `doc_type`s equally.

### `QdrantManager` additions (`app/database/qdrant_client.py`)

```python
async def _create_personal_collection(self) -> None: ...
async def ensure_personal_collection(self) -> None: ...
```

Called once during FastAPI lifespan startup. Idempotent. Same dimension-mismatch protection as existing `ensure_kb_collection()`.

### Document `metadata_` schema

```json
{
  "doc_type": "homepage" | "project" | "cv",
  "project_slug": "<slug>",        // project only, e.g. "agent-network"
  "project_url": "<full-url>",     // project only
  "scraped_at": "<iso8601>",
  "source": "<filename or URL>"
}
```

---

## 4. Portfolio Scraper — `app/ingestion/portfolio_sync.py`

### Pipeline

1. Fetch `sitemap.xml` → parse XML → filter URLs:
   - Homepage: exact match `portfolio_homepage_url`.
   - Projects: regex match `portfolio_project_url_pattern` (expected: 10 URLs).
   - CV: hardcoded `portfolio_cv_url` (PDFs not always in sitemap).
2. **For each web URL** (homepage + 10 projects):
   1. `httpx.GET` with realistic User-Agent.
   2. Parse HTML with `BeautifulSoup4`.
   3. Extract `<main>` or `<article>`; fallback to `<body>`.
   4. Convert HTML → Markdown via `markdownify`.
   5. Strip noise (`<nav>`, `<footer>`, `<script>`, `<style>`, comments).
   6. Compute SHA-256 content hash → skip if unchanged (unless `force_reingest`).
3. **For CV PDF**:
   1. `httpx.GET` binary.
   2. Parse with `pypdf` — extract text per page, joined with `\n\n## Page N\n\n` markers.
   3. Same hash check.
4. **Per document**:
   - Build metadata (`doc_type`, `project_slug`, `project_url`, `scraped_at`, `source`).
   - Use `MarkdownNodeParser` to chunk by headers.
   - Re-split oversized (>600 token) sections via configured `TokenTextSplitter` (defense-in-depth).
   - Filter empty/whitespace-only nodes.
   - Embed + upsert to `Personal_Portfolio` via `LlamaIndex.ainsert_nodes`.
   - Persist `Document` + `Chunk` records to PostgreSQL.
5. **Stale cleanup**: delete from Qdrant + PostgreSQL any document whose `source`/`project_url` no longer appears in the latest sitemap scrape (mirrors Moodle sync's `_delete_stale_documents`).

### Helper functions (private)

```python
async def _fetch_sitemap_urls(client) -> dict
    # Returns {"homepage": str, "projects": list[str], "cv": str}

async def _scrape_web_page(client, url) -> tuple[str, str]
    # Returns (markdown_content, page_title)

async def _scrape_cv_pdf(client, url) -> tuple[str, str]
    # Returns (markdown_content, "Curriculum Vitae")

def _build_metadata(doc_type, url, scraped_at) -> dict
def _slugify_project_url(url) -> str

async def _ingest_portfolio_doc(
    raw_markdown, source_id, title, metadata,
    session, force_reingest
) -> int

async def _delete_stale_portfolio_docs(session, current_sources)
```

### Public entrypoint

```python
async def sync_portfolio_knowledge_base(
    session: AsyncSession,
    force_reingest: bool = False,
) -> dict[str, Any]:
    """
    Returns: {"docs_processed", "chunks_ingested", "docs_skipped", "errors"}
    """
```

### arq worker registration (`app/worker.py`)

```python
async def sync_portfolio_task(ctx, force_reingest: bool = False):
    async with AsyncSessionLocal() as session:
        return await sync_portfolio_knowledge_base(session, force_reingest)
```

Register in worker's `WorkerSettings.functions` list.

### New dependencies (`pyproject.toml`)

```toml
beautifulsoup4 = "^4.12"
markdownify = "^0.13"
pypdf = "^4.3"
```

### Robustness

- `httpx` timeout 30s per URL.
- Per-URL `try/except` → one failed page does not abort sync; error appended to `summary["errors"]`.
- User-Agent: `"Mozilla/5.0 (compatible) AskferBot/1.0"` to avoid bot blocks.
- Title fallback: use HTML `<title>` tag if no `<h1>` present.
- Sitemap fetch failure: fall back to hardcoded URL list (homepage + CV; projects empty → log warning).

---

## 5. Askfer API Route + IP Rate Limiting

### New file: `app/api/askfer_deps.py`

```python
async def rate_limit_by_ip(request: Request) -> str:
    """Public endpoint guard: rate-limit by client IP. No auth."""
    fwd = request.headers.get("x-forwarded-for", "")
    ip = fwd.split(",")[0].strip() if fwd else (
        request.client.host if request.client else "unknown"
    )

    redis = get_redis_client()
    key = f"askfer:rate:{ip}"
    limit = settings.askfer_rate_limit_per_minute

    try:
        pipe = redis.pipeline()
        pipe.incr(key)
        pipe.expire(key, 60)
        results = await pipe.execute()
        if results[0] > limit:
            raise HTTPException(429, f"Rate limit exceeded ({limit}/min)")
    except HTTPException:
        raise
    except Exception as exc:
        # Fail-open if Redis hiccups (mirrors auth.py pattern).
        logger.warning(f"Rate limit Redis error (allowing request): {exc}")
    return ip
```

### Schemas (append to `app/api/schemas.py`)

```python
class AskferRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)

class AskferSyncRequest(BaseModel):
    force_reingest: bool = False
```

No `conversation_id` — Askfer is stateless.

### New file: `app/api/routes/askfer.py`

```python
router = APIRouter()

@router.post("/askfer/stream", summary="Ask Askfer about Ferdy's portfolio (SSE)")
async def askfer_stream(
    request: AskferRequest,
    req: Request,
    db: AsyncSession = Depends(get_db),
    client_ip: str = Depends(rate_limit_by_ip),
):
    """Mirrors /chat/stream structure. Uses askfer_pipeline + Personal_Portfolio.
    No auth, no conversation history, no LTM."""
    ...

@router.post("/askfer/sync", summary="Trigger portfolio re-sync (admin)")
async def askfer_sync(
    request: AskferSyncRequest,
    x_admin_secret: str = Header(...),
):
    if not settings.askfer_admin_secret or not secrets.compare_digest(
        x_admin_secret, settings.askfer_admin_secret
    ):
        raise HTTPException(403, "Forbidden")
    arq_redis = await get_arq_redis()
    job = await arq_redis.enqueue_job('sync_portfolio_task', request.force_reingest)
    return {"message": "Portfolio sync enqueued", "job_id": job.job_id}
```

`secrets.compare_digest()` is used to avoid timing attacks on the admin secret check.

### Mounting (`app/main.py`)

```python
from app.api.routes import chat, ingest, askfer  # add askfer
...
app.include_router(askfer.router, prefix="/api/v1", tags=["Askfer"])
```

### Lifespan addition (`app/main.py`)

```python
await qdrant.ensure_personal_collection()
logger.info("Qdrant Personal_Portfolio collection ready")
```

### Endpoint summary

| Endpoint | Method | Auth | Rate limit |
|---|---|---|---|
| `/api/v1/askfer/stream` | POST | none | 10/min per IP |
| `/api/v1/askfer/sync` | POST | `X-Admin-Secret` header | n/a |

### CORS

Existing `allow_origins=["*"]` works for MVP. Optional later lockdown:
```python
allow_origins=["https://ferdy-fadhil-lazuardi.my.id", "http://localhost:*"]
```

---

## 6. Askfer Pipeline (Graph + Persona)

### New file: `app/llm/askfer_prompts.py`

```python
ASKFER_PERSONA = (
    "You are Askfer, an AI assistant representing Ferdy Fadhil Lazuardi. "
    "You speak in FIRST-PERSON as Ferdy himself ('saya' / 'I'). "
    "You introduce, explain, and answer questions about Ferdy's projects, "
    "skills, and professional background to recruiters, HR, and visitors of "
    "Ferdy's portfolio website.\n\n"
    "LANGUAGE — bilingual auto-detect:\n"
    "- Default: English (most visitors are international recruiters).\n"
    "- If the user's message is in Indonesian, switch to Indonesian using "
    "  casual-professional 'saya/kamu'.\n"
    "- Never mix languages within one response.\n\n"
    "TONE — professional-casual:\n"
    "- Confident, friendly, concise. Like talking to a recruiter over coffee.\n"
    "- Open with the answer. No filler ('Sure!', 'Of course!', 'Tentu!').\n"
    "- End with substance. No 'Hope that helps!', 'Semoga membantu!'.\n"
    "- No hedging: 'maybe', 'I think', 'mungkin', 'sepertinya'.\n"
    "- Use complete sentences; bullets for lists of features/tech/responsibilities.\n"
    "- Preserve project names, tech stack names, percentages, and numbers verbatim."
)

ASKFER_SYSTEM_PROMPT = f"""<role>
{ASKFER_PERSONA}
</role>

<rules>
1. Answer ONLY using <retrieved_context>. Never fabricate projects, tech stacks, dates, or roles.
2. NOT FOUND: respond honestly + redirect to contact.
   - EN: "I don't have that detail in my portfolio yet. For deeper questions, reach me on LinkedIn or email — links are on my homepage."
   - ID: "Detail itu belum ada di portfolio-ku. Buat pertanyaan lebih lanjut, kontak aku via LinkedIn atau email — link-nya ada di homepage."
3. SCOPE — only answer about: projects, tech stack, professional experience, skills, education, contact info.
   Off-scope (salary expectations, opinions on other people, personal life, politics) → polite redirect:
   "I keep this chat focused on my professional work. For other things, reach out directly."
4. FOLLOW-UPS — same grounding rule as A-Pedi:
   - 0 to 3 follow-ups, ONLY if the answer is directly present in <retrieved_context>.
   - If context supports zero, omit the block entirely.

Format:
[direct answer in first-person]

**Curious about:**   (or **Penasaran tentang:** if responding in Indonesian)
1. [grounded follow-up]
2. [grounded follow-up]
</rules>"""
```

### New file: `app/graph/askfer_pipeline.py`

Reuses existing `RAGState` (`app/graph/state.py`) — only `messages`, `retrieved_context`, `intent`, `rewritten_query` fields are populated. `conversation_id`, `conversation_summary`, `user_profile`, `user_preferences` left empty/None.

### Graph topology

```
START
  ↓
pre_processor      ← classify intent (GREETING / OFF_SCOPE / MALICIOUS / KNOWLEDGE)
  ↓ (route by intent)
  ├─ GREETING       → handle_greeting    → END
  ├─ OFF_SCOPE      → handle_off_scope   → END
  ├─ MALICIOUS      → handle_malicious   → END
  └─ KNOWLEDGE      → rag_node → generate_node → END
```

### Differences vs A-Pedi pre_processor

- Intent set: `GREETING | OFF_SCOPE | MALICIOUS | KNOWLEDGE` (replaces A-Pedi's `AMBIGUOUS` with `OFF_SCOPE` — recruiter context rarely needs clarification but often goes off-topic).
- Stateless: `rewritten_query` = original query (no history rewrite).
- Heuristic short-circuit for clear greetings (zero-LLM path) — same idea as A-Pedi.

### `_rag_node` (Askfer)

```python
docs = await hybrid_search(
    query=query_to_search,
    collection=settings.qdrant_personal_collection,  # ← key difference
)
reranked = await rerank(query=query_to_search, chunks=docs)
```

### `_generate_node` (Askfer)

- System prompt = `ASKFER_SYSTEM_PROMPT + "\n\n<retrieved_context>\n...\n</retrieved_context>"`.
- No LTM section, no summary section, no `user_preferences` section.
- Apply existing `validate_followups` + `render_followup_block` (collection-agnostic).

### Singleton

```python
@lru_cache(maxsize=1)
def get_askfer_graph():
    return _build_askfer_graph()
```

### Estimated LoC

- `askfer_pipeline.py`: ~200 LoC.
- `askfer_prompts.py`: ~50 LoC.

---

## 7. Frontend Integration & Cache Strategy

### Streaming endpoint flow

```python
@router.post("/askfer/stream")
async def askfer_stream(request, req, db, client_ip):
    start_time = time.perf_counter()

    # 1. Cache lookup (namespace="askfer")
    query_embedding = await LISettings.embed_model.aget_query_embedding(request.query)
    cached = await get_cached_response(
        request.query,
        course_id=None,
        query_embedding=query_embedding,
        cache_namespace="askfer",
    )
    if cached:
        return StreamingResponse(_stream_cached(cached), media_type="text/event-stream", ...)

    # 2. Initial state — minimal, stateless
    initial_state = {"messages": [HumanMessage(content=request.query)]}

    # 3. Stream via askfer graph
    askfer_graph = get_askfer_graph()
    return StreamingResponse(_stream_askfer(...), media_type="text/event-stream", ...)
```

### Cache namespace isolation (only edit to existing code)

The current `app/utils/cache.py` uses a single Redis prefix (`rag:cache:`) and a single shared Qdrant collection (`semantic_cache`) for both exact and semantic matches, scoped only by `course_id`. Naive reuse would let Askfer queries semantically match A-Pedi cached answers (and vice versa). Both layers need namespace isolation.

**`app/utils/cache.py` changes (additive, default preserves A-Pedi):**

1. Add a `cache_namespace` parameter to `get_cached_response`, `set_cached_response`, `flush_cache_by_course`, and `flush_cache`. Default literal is `"rag"` to keep Redis key prefix identical to today (`rag:cache:...`):

   ```python
   async def get_cached_response(
       query: str,
       course_id: int | None = None,
       query_embedding: list[float] | None = None,
       cache_namespace: str = "rag",
   ) -> dict | None: ...

   async def set_cached_response(
       query: str,
       answer: str,
       sources: list[dict],
       course_id: int | None = None,
       ttl: int | None = None,
       query_embedding: list[float] | None = None,
       cache_namespace: str = "rag",
   ) -> None: ...
   ```

2. Replace the module-level `_PREFIX = "rag:cache:"` with a helper:

   ```python
   def _cache_key(query, course_id, namespace="rag"):
       query_hash = hashlib.sha256(query.strip().lower().encode()).hexdigest()[:16]
       cid_str = str(course_id) if course_id and course_id > 0 else "global"
       return f"{namespace}:cache:{cid_str}:{query_hash}"
   ```

   With default `namespace="rag"`, A-Pedi keys remain byte-identical to current production. Askfer passes `cache_namespace="askfer"` → keys prefixed `askfer:cache:...`.

3. Persist the namespace into the Qdrant `semantic_cache` payload, and filter by it on read:

   ```python
   payload = {"answer": answer, "sources": sources, "namespace": cache_namespace}

   # In get_cached_response, build query_filter:
   must_clauses = [
       qdrant_models.FieldCondition(
           key="namespace",
           match=qdrant_models.MatchValue(value=cache_namespace),
       )
   ]
   if course_id is not None and course_id > 0:
       must_clauses.append(qdrant_models.FieldCondition(
           key="course_id",
           match=qdrant_models.MatchValue(value=course_id),
       ))
   query_filter = qdrant_models.Filter(must=must_clauses)
   ```

4. Add a payload index on `namespace` (KEYWORD) inside `_ensure_semantic_collection()` so the filter is fast.

5. **Backfill consideration:** existing `semantic_cache` points (written before this change) lack a `namespace` field, so the new filter excludes them. That's acceptable — those entries simply expire via TTL or the next cache rewrite. No migration required. Redis exact-match cache is unaffected (only the prefix changes for Askfer's keys, which start fresh).

A-Pedi runtime behavior is unchanged: same default namespace, same Redis prefix, same Qdrant collection. Only `cache.py` itself is edited; A-Pedi call sites are not modified.

### SSE event protocol

```
data: {"token": "Hi! "}
data: {"token": "I'm Ferdy..."}
event: done
data: {"sources": [{...}], "cached": false, "latency_ms": 1234.5}
```

No `event: resolved` (no rewrite for stateless).

### Source citations format

```python
def _extract_askfer_sources(retrieved_context: list) -> list:
    sources = []
    seen = set()  # dedupe by project_url / source
    for c in retrieved_context:
        meta = c.get("metadata") or {}
        url = meta.get("project_url") or c.get("source", "")
        if url in seen:
            continue
        seen.add(url)
        sources.append({
            "doc_type": meta.get("doc_type", ""),
            "title": c.get("course_name") or c.get("title") or "Unknown",
            "url": url,
            "score": c.get("score") or 0.0,
        })
    return sources
```

Dedupe ensures one project surfaces as one source even though it has multiple chunks.

### Frontend deliverables

- `app/static/askfer.html` — minimal dev demo widget (mirror of `app/static/index.html`, points to `/api/v1/askfer/stream`).
- This spec doc serves as the API contract for the portfolio website integration.

---

## 8. Observability, Testing, & Rollout

### Langfuse tracing

```python
with propagate_attributes(
    trace_name="askfer-stream",
    session_id=f"ip:{client_ip}",
    user_id=f"anon-{client_ip}",
    tags=["askfer", settings.app_env],
    metadata={
        "collection": "Personal_Portfolio",
        "streaming": True,
        "query": request.query[:80],
    },
):
    ...
```

Filtering `tags:askfer` in the Langfuse dashboard separates Askfer traffic from A-Pedi (`tags:api-chat`). Retrieval scores (`retriever_hybrid_max`, `retriever_hybrid_avg`, `retriever_cohere_max`, `retriever_cohere_avg`) are submitted using the same pattern as A-Pedi.

### BatchLogger

`batch_logger.add_log({...})` is reused; add field `"endpoint": "askfer"` for downstream filtering.

### Tests (`tests/`)

```
tests/
├── test_askfer_route.py
│   ├── test_stream_no_auth_returns_200
│   ├── test_stream_rate_limit_per_ip
│   ├── test_sync_requires_admin_secret
│   ├── test_sync_constant_time_compare
│   └── test_no_jwt_required
├── test_portfolio_sync.py        # mocked httpx
│   ├── test_sitemap_parse_filters_project_urls
│   ├── test_homepage_scraped_as_homepage_doctype
│   ├── test_cv_pdf_text_extracted
│   ├── test_skips_unchanged_content_hash
│   └── test_stale_doc_cleanup
├── test_askfer_pipeline.py       # mocked LLM + retriever
│   ├── test_persona_first_person_response
│   ├── test_off_scope_redirects_to_contact
│   ├── test_not_found_response_bilingual
│   ├── test_collection_param_personal_portfolio
│   └── test_no_conversation_history_used
└── test_apedi_unaffected.py      # regression
    ├── test_chat_route_still_uses_knowledge_base_collection
    ├── test_apedi_persona_unchanged
    └── test_apedi_cache_namespace_default_unchanged
```

`pytest-asyncio` already in dev deps (verify in `pyproject.toml`; add if missing).

### Rollout order

1. Add settings → no behavior change.
2. Add `ensure_personal_collection()` to lifespan → idempotent.
3. Add `cache_namespace` param to `cache.py` (default preserves A-Pedi keys) → only edit to existing code.
4. Add scraper + worker task → standalone, no A-Pedi touch.
5. First sync via `POST /askfer/sync` → populate `Personal_Portfolio`.
6. Add Askfer routes + pipeline + prompts → fully isolated additions.
7. Smoke test: A-Pedi (`/chat/stream` against `Knowledge_Base`) + Askfer (`/askfer/stream` against `Personal_Portfolio`).
8. Optional: tighten CORS to portfolio domain.

### Failure modes & handling

| Failure | Handling |
|---|---|
| `sitemap.xml` unreachable | Fall back to hardcoded URL list (homepage + CV; projects empty → log warning). |
| CV PDF download fails | Log warning, skip CV, continue with project + homepage. |
| Single project page 404 | Log error, skip that page, continue. |
| `Personal_Portfolio` empty at first query | Honest "I don't have that detail yet…" response (graceful degradation). |
| Rate-limit Redis fails | Fail-open: log warning, allow request (mirrors `auth.py`). |

### Acceptance criteria (DoD)

1. `POST /api/v1/askfer/stream` returns SSE stream answering project/CV/homepage questions in first-person.
2. `POST /api/v1/askfer/sync` ingests homepage + 10 projects + CV PDF into `Personal_Portfolio`.
3. Existing `/api/v1/chat` & `/api/v1/chat/stream` regression tests pass.
4. Rate limit by IP enforced (16th request within 1 minute → 429).
5. Off-scope query → polite redirect; not-found query → honest "tanya via LinkedIn/email".
6. Bilingual auto-detect verified ("What's your tech stack?" → EN; "Apa tech stack kamu?" → ID).
7. Source citations dedupe per project URL.

---

## 9. Out of scope

- Frontend chat widget on `ferdy-fadhil-lazuardi.my.id` (consumer integrates separately).
- Auto-scheduled cron re-sync (manual trigger only for MVP).
- Multi-language beyond ID/EN.
- Long-term memory / persistent user profiles.
- Abuse detection beyond IP rate limiting (e.g., abusive query classification).
- CORS origin lockdown (kept open in MVP; documented as optional follow-up).

---

## 10. New files / modified files summary

### New files

```
app/api/askfer_deps.py              # rate_limit_by_ip dependency
app/api/routes/askfer.py            # /askfer/stream, /askfer/sync endpoints
app/graph/askfer_pipeline.py        # Askfer LangGraph pipeline
app/llm/askfer_prompts.py           # ASKFER_PERSONA, ASKFER_SYSTEM_PROMPT
app/ingestion/portfolio_sync.py     # sitemap + project + CV scraper
app/static/askfer.html              # dev test UI
tests/test_askfer_route.py
tests/test_portfolio_sync.py
tests/test_askfer_pipeline.py
tests/test_apedi_unaffected.py
```

### Modified files (additive only)

```
app/config/settings.py              # 7 new settings fields
app/database/qdrant_client.py       # _create_personal_collection, ensure_personal_collection
app/api/schemas.py                  # AskferRequest, AskferSyncRequest
app/main.py                         # include_router(askfer.router), ensure_personal_collection() in lifespan
app/worker.py                       # register sync_portfolio_task
app/utils/cache.py                  # add optional cache_namespace param (default preserves A-Pedi)
pyproject.toml                      # beautifulsoup4, markdownify, pypdf
.env.example                        # ASKFER_ADMIN_SECRET placeholder
```

A-Pedi runtime files (`app/graph/pipeline.py`, `app/api/routes/chat.py`, `app/llm/prompts.py`) are **not** modified.
