"""
Portfolio → Qdrant `Personal_Portfolio` Synchronisation Script (Askfer)
========================================================================
1. Fetches sitemap.xml from the portfolio site to discover URLs.
2. Filters URLs into homepage, project pages, and CV PDF.
3. Scrapes each web page → HTML → cleaned Markdown (BeautifulSoup + markdownify).
4. Downloads CV PDF → extracts text per page (pypdf).
5. Computes content_hash to skip unchanged docs (unless force_reingest).
6. Splits content via MarkdownNodeParser; re-splits oversized header sections.
7. Embeds + upserts to Qdrant `Personal_Portfolio` collection (hybrid dense+sparse).
8. Persists Document + Chunk records to PostgreSQL.
9. Stale cleanup: deletes docs whose URL is no longer in the latest scrape.
"""
import datetime
import hashlib
import re
import uuid
from typing import Any, cast

import httpx
from loguru import logger
from sqlalchemy import delete, or_, select
from sqlalchemy.sql import text as sql_text
from sqlalchemy.ext.asyncio import AsyncSession
from qdrant_client import models as qdrant_models

from app.config.settings import get_settings
from app.config.embedding_config import ensure_llamaindex_configured
from app.database.models import Chunk, Document
from app.database.qdrant_client import get_qdrant_client
from app.utils.token_counter import count_tokens

from llama_index.core import Document as LlamaDocument, VectorStoreIndex, StorageContext
from llama_index.core.schema import BaseNode
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core import Settings as LISettings

settings = get_settings()

USER_AGENT = "Mozilla/5.0 (compatible) AskferBot/1.0 (+https://ferdy-fadhil-lazuardi.my.id)"

# Editable profile file — single source of truth for "who is Ferdy" answers.
# Reread on every sync; rebuilt as one chunk so retrieval surfaces it whole.
PROFILE_MD_PATH = "data/personal/profile.md"

# Local mirror of scraped sources. Sync flow is local-first: when a file exists
# here, it is used verbatim and the website is NOT touched. When a file is
# absent, the scraper writes its output here so subsequent syncs (and Proxmox
# deploys) work fully offline. Edit any of these to override what Askfer says.
LOCAL_HOMEPAGE_PATH = "data/personal/homepage.md"
LOCAL_CV_PATH = "data/personal/cv.md"
LOCAL_PROJECTS_DIR = "data/personal/projects"
LOCAL_KNOWLEDGE_DIR = "data/personal/knowledge"


def _read_local_md(path: str) -> str | None:
    """Read a markdown file from disk. Returns None if missing/empty/error."""
    import os
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        return content or None
    except Exception as exc:
        logger.warning("Local md read failed", path=path, error=str(exc))
        return None


def _write_local_md(path: str, content: str) -> None:
    """Write markdown to disk. Creates parent dir as needed."""
    import os
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content.strip() + "\n")
    except Exception as exc:
        logger.warning("Local md write failed", path=path, error=str(exc))


def _load_local_with_title(path: str, default_title: str) -> tuple[str, str] | None:
    """Load a local md file and extract its H1 as title.

    Returns (markdown, title) or None if missing.
    """
    md = _read_local_md(path)
    if not md:
        return None
    title = default_title
    for line in md.splitlines():
        s = line.strip()
        if s.startswith("# "):
            title = s.lstrip("# ").strip() or default_title
            break
    return md, title


def _save_local_with_title(path: str, md: str, title: str) -> None:
    """Save md to disk, prepending an H1 title if not already present."""
    has_h1 = any(line.strip().startswith("# ") for line in md.splitlines()[:5])
    body = md if has_h1 else f"# {title}\n\n{md}"
    _write_local_md(path, body)


# UI/widget noise lines that markdownify converts into standalone paragraphs.
# Stripped before chunking — they waste tokens and don't help retrieval.
NOISE_LINES = {
    "Module Progress0%", "SCORM Ready", "Exported successfully",
    "Storyboarding", "In progress...", "Interactive_Module.mp4",
    "Interactive\\_Module.mp4", "Featured", "Coming soon",
    "BACK.", "Back toPROJECTS.",
}


def _clean_markdown(md: str) -> str:
    """Strip noise from scraped markdown — images, UI widget labels, and
    consecutive duplicate lines (logo carousels, marquees)."""
    if not md:
        return md
    # Drop all markdown images (logos, screenshots — useless for text RAG)
    md = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", md)
    # Drop bare anchor-only links like [BACK.](/path) that markdownify leaves behind
    md = re.sub(r"^\[[^\]]*\]\([^)]+\)\s*$", "", md, flags=re.MULTILINE)

    out_lines: list[str] = []
    prev_stripped: str | None = None
    for line in md.splitlines():
        stripped = line.strip()
        # Skip known noise lines
        if stripped in NOISE_LINES:
            continue
        # Collapse consecutive duplicate non-empty lines (carousels)
        if stripped and stripped == prev_stripped:
            continue
        out_lines.append(line)
        if stripped:
            prev_stripped = stripped

    cleaned = "\n".join(out_lines)
    # Collapse 3+ blank lines into 2
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


# Static counter values — homepage shows JS-animated counters that the SSR
# baseline renders as `0`. Patch them in post-scrape so Askfer answers with
# the real numbers. Update here when the portfolio site updates.
HOMEPAGE_COUNTER_OVERRIDES: list[tuple[str, str]] = [
    ("Users Empowered", "10.000+"),
    ("Completion Rate", "65%"),
    ("Satisfaction", "3.64/4"),
    ("Years Experience", "2+"),
]


def _patch_homepage_counters(markdown: str) -> str:
    """Replace zero-placeholders adjacent to the known counter labels.

    The site SSRs each stat as a literal `0` (with trailing `+`, `%`, or `/4`)
    just before the label, then animates it on hydrate. After markdownify the
    pattern looks like:
        0+

        Users Empowered

    For each label, find the *nearest preceding* line that matches one of
    `0+`, `0%`, `0/4` and substitute it with the real value.
    """
    if not markdown:
        return markdown
    lines = markdown.splitlines()
    label_to_value = {label: value for label, value in HOMEPAGE_COUNTER_OVERRIDES}
    placeholder_re = re.compile(r"^\s*0([+%]|/\d+)\s*$")

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped not in label_to_value:
            continue
        # Walk backward to find the nearest placeholder line
        for j in range(i - 1, max(-1, i - 6), -1):
            if placeholder_re.match(lines[j]):
                lines[j] = label_to_value[stripped]
                break
    return "\n".join(lines)


# ─── Sitemap discovery ───────────────────────────────────────────────────────

async def _fetch_sitemap_urls(client: httpx.AsyncClient) -> dict:
    """
    Fetch sitemap.xml and partition URLs into {homepage, projects, cv}.
    Falls back to hardcoded URLs (homepage + CV) if sitemap is unreachable.
    Returns: {"homepage": str, "projects": list[str], "cv": str}
    """
    fallback = {
        "homepage": settings.portfolio_homepage_url,
        "projects": [],
        "cv": settings.portfolio_cv_url,
    }
    try:
        resp = await client.get(settings.portfolio_sitemap_url, headers={"User-Agent": USER_AGENT})
        resp.raise_for_status()
        xml = resp.text
    except Exception as exc:
        logger.warning("Sitemap fetch failed — falling back to hardcoded URLs", error=str(exc))
        return fallback

    # Parse <loc>...</loc> entries naively (avoids ElementTree namespace pain).
    urls = re.findall(r"<loc>\s*([^<\s]+)\s*</loc>", xml)
    homepage = settings.portfolio_homepage_url.rstrip("/") + "/"
    project_pattern = re.compile(settings.portfolio_project_url_pattern)

    result: dict[str, Any] = {"homepage": homepage, "projects": [], "cv": settings.portfolio_cv_url}
    seen_projects: set[str] = set()
    for u in urls:
        u_norm = u.strip()
        if u_norm in seen_projects:
            continue
        if u_norm.rstrip("/") + "/" == homepage:
            result["homepage"] = u_norm
        elif project_pattern.match(u_norm):
            result["projects"].append(u_norm)
            seen_projects.add(u_norm)

    logger.info(
        "Sitemap parsed",
        homepage=result["homepage"],
        projects_found=len(result["projects"]),
        cv=result["cv"],
    )
    return result


# ─── Web page scraping ───────────────────────────────────────────────────────

async def _scrape_web_page(client: httpx.AsyncClient, url: str) -> tuple[str, str]:
    """Download a web page, clean, and convert to Markdown.
    Returns: (markdown_content, page_title)
    """
    from bs4 import BeautifulSoup, Comment
    from markdownify import markdownify as md

    resp = await client.get(url, headers={"User-Agent": USER_AGENT}, follow_redirects=True)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    # Title: prefer <h1>, fall back to <title>
    title = ""
    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        title = h1.get_text(strip=True)
    elif soup.title and soup.title.string:
        title = soup.title.string.strip()

    # Strip noise
    for tag in soup(["script", "style", "noscript", "nav", "footer", "header", "aside", "form"]):
        tag.decompose()
    for c in soup.find_all(string=lambda s: isinstance(s, Comment)):
        c.extract()

    # Pick main content area
    main = soup.find("main") or soup.find("article") or soup.body or soup
    html = str(main)

    markdown = md(html, heading_style="ATX", bullets="-", strip=["script", "style"])
    markdown = _clean_markdown(markdown)

    return markdown, title or url


# ─── CV PDF parsing ──────────────────────────────────────────────────────────

async def _scrape_cv_pdf(client: httpx.AsyncClient, url: str) -> tuple[str, str]:
    """Download a PDF and extract text per page as Markdown.
    Returns: (markdown_content, "Curriculum Vitae")
    """
    import io
    from pypdf import PdfReader

    resp = await client.get(url, headers={"User-Agent": USER_AGENT}, follow_redirects=True)
    resp.raise_for_status()
    reader = PdfReader(io.BytesIO(resp.content))

    parts = ["# Curriculum Vitae — Ferdy Fadhil Lazuardi\n"]
    for i, page in enumerate(reader.pages, 1):
        try:
            page_text = page.extract_text() or ""
        except Exception as exc:
            logger.warning(f"PDF page {i} extract failed", error=str(exc))
            page_text = ""
        page_text = page_text.strip()
        if not page_text:
            continue
        parts.append(f"\n## Page {i}\n\n{page_text}\n")

    markdown = "\n".join(parts).strip()
    return markdown, "Curriculum Vitae"


# ─── Helpers ────────────────────────────────────────────────────────────────

def _slugify_project_url(url: str) -> str:
    """Extract <slug> from .../projects/<slug>/."""
    m = re.search(r"/projects/([^/]+)/?$", url.rstrip("/") + "/")
    return m.group(1) if m else url


def _build_metadata(doc_type: str, url: str, scraped_at: str, project_slug: str = "") -> dict:
    meta = {
        "doc_type": doc_type,
        "scraped_at": scraped_at,
        "source": url,
    }
    if doc_type == "project":
        meta["project_slug"] = project_slug or _slugify_project_url(url)
        meta["project_url"] = url
    return meta


def _load_profile_markdown() -> tuple[str, str] | None:
    """Read the editable profile.md from disk. Returns (markdown, title) or
    None when the file is absent (sync remains successful — profile is optional)."""
    import os

    if not os.path.exists(PROFILE_MD_PATH):
        logger.info("profile.md not found, skipping profile ingestion", path=PROFILE_MD_PATH)
        return None
    try:
        with open(PROFILE_MD_PATH, "r", encoding="utf-8") as f:
            md = f.read().strip()
    except Exception as exc:
        logger.warning("Failed to read profile.md", error=str(exc))
        return None
    if not md:
        logger.info("profile.md is empty, skipping")
        return None

    # Title = first H1 if present, else default
    title = "About Ferdy Fadhil Lazuardi"
    for line in md.splitlines():
        s = line.strip()
        if s.startswith("# "):
            title = s.lstrip("# ").strip() or title
            break
    return md, title


# ─── Overview generation (LLM-extracted org labels + one-liners) ────────────

# Static slug → org mapping. LLM extraction can't reliably know
# "Dunia Geometri = Unnes" from page content alone, so we hardcode the truth.
# Update here when a project moves orgs or a new project is added.
PROJECT_ORG_OVERRIDES: dict[str, str] = {
    "amartha-lms-chatbot": "Amartha",
    "agent-network": "Amartha",
    "training-client-protection": "Amartha",
    "anti-harassment": "Amartha",
    "modal": "Amartha",
    "asa": "Amartha",
    "amarthafin-mockup": "Amartha",
    "dunia-geometri": "Unnes",
    "bts": "BPTIK DIKBUD Jateng",
    "botani-quest": "BPTIK DIKBUD Jateng",
}

# Per-org context line shown in the overview doc so the LLM understands the
# relationship (employment, academic collaboration, etc.) and can answer
# follow-up questions like "is the Unnes project personal?" correctly.
ORG_CONTEXT: dict[str, str] = {
    "Amartha": "Built during my paid internship at Amartha (Digital Learning team). NOT a full-time role.",
    "Unnes": "Paid freelance work commissioned by my lecturer at Universitas Negeri Semarang to develop interactive learning media for elementary schools — distributed and used by partner schools. NOT a thesis / tugas akhir / skripsi project.",
    "BPTIK DIKBUD Jateng": "Built during my internship at BPTIK Dinas Pendidikan Provinsi Jawa Tengah. Outputs were distributed to schools across Central Java. NOT freelance / commissioned / paid contract work.",
}


async def _summarize_project_for_overview(markdown: str, project_url: str, title: str) -> dict:
    """Cheap-LLM extract: organization label + one-line summary for the overview doc.

    `org_label` is overridden from the static `PROJECT_ORG_OVERRIDES` map when
    the slug is known — LLM extraction is only used as a fallback for new
    projects we haven't mapped yet.
    """
    from langchain_core.messages import HumanMessage
    from pydantic import BaseModel, Field

    from app.llm.client import get_cheap_llm

    class ProjectSummary(BaseModel):
        org_label: str = Field(
            description=(
                "The organization or client this project was done for. "
                "Pick the most prominent name mentioned (e.g., 'Amartha', "
                "'Unnes', 'BPTIK', 'Skilvul', 'Binar'). If none mentioned, "
                "use 'Personal'. Single short label, no extra words."
            )
        )
        one_liner: str = Field(
            description=(
                "One-sentence first-person summary of what I built/did in this project. "
                "Max 22 words. No filler. Mention the medium (e.g., interactive module, "
                "video, mockup) and primary outcome."
            )
        )

    snippet = (markdown or "")[:2500]
    prompt = (
        f"Project URL: {project_url}\n"
        f"Project title: {title}\n\n"
        f"Project content:\n{snippet}\n\n"
        "Extract org_label and one_liner."
    )
    llm = get_cheap_llm()
    structured = llm.with_structured_output(ProjectSummary)
    result = cast(ProjectSummary, await structured.ainvoke([HumanMessage(content=prompt)]))

    slug = _slugify_project_url(project_url)
    org_from_map = PROJECT_ORG_OVERRIDES.get(slug)
    org_label = org_from_map or (result.org_label or "Personal").strip() or "Personal"

    return {
        "title": title or slug,
        "url": project_url,
        "org_label": org_label,
        "one_liner": (result.one_liner or "").strip(),
    }


def _build_overview_markdown(summaries: list[dict]) -> str:
    """Render compact markdown index of ALL projects, grouped by org_label.

    Uses ONE H1 + bold inline labels (no H2/H3) so MarkdownNodeParser keeps
    everything in a single chunk — guarantees retrieval surfaces the complete
    list when the user asks "what projects have you done?"

    Each org header includes a short context line from `ORG_CONTEXT` so the
    LLM can correctly classify the relationship (employment vs. collaboration
    vs. personal) when asked.
    """
    from collections import defaultdict

    if not summaries:
        return ""

    by_org: dict[str, list[dict]] = defaultdict(list)
    for s in summaries:
        by_org[s["org_label"] or "Personal"].append(s)

    lines = [
        "# Portfolio Overview — All Projects",
        "",
        "Daftar lengkap proyek saya / Full list of my projects, grouped by organization or client.",
        "Note on engagement type (NONE of these are personal/skripsi, NONE are full-time roles): Amartha projects = built during my PAID INTERNSHIP at Amartha. Unnes projects = PAID FREELANCE work commissioned by my lecturer. BPTIK DIKBUD Jateng projects = built during my INTERNSHIP there. Use the exact engagement label per org — do NOT mix them up, and do NOT call any of these full-time roles.",
        "",
    ]
    # Stable order: Amartha first, then alpha. "Personal" sinks to the bottom.
    def _key(label: str) -> tuple:
        low = label.lower()
        return (low != "amartha", low == "personal", low)

    for org in sorted(by_org.keys(), key=_key):
        items = by_org[org]
        lines.append(f"**{org}:**")
        ctx = ORG_CONTEXT.get(org)
        if ctx:
            lines.append(f"_{ctx}_")
        for s in items:
            lines.append(f"- **{s['title']}** — {s['one_liner']} ({s['url']})")
        lines.append("")
    return "\n".join(lines).strip()


# ─── Stale cleanup ──────────────────────────────────────────────────────────

async def _delete_stale_portfolio_docs(session: AsyncSession, current_sources: list[str]):
    """Delete docs in PG/Qdrant whose source isn't in `current_sources`."""
    stmt = select(Document).where(
        sql_text("metadata->>'doc_type' IN ('homepage', 'project', 'cv', 'profile', 'overview', 'knowledge')")
    )
    result = await session.execute(stmt)
    docs = result.scalars().all()

    stale_docs = [d for d in docs if d.source not in current_sources]
    if not stale_docs:
        return

    stale_ids = [str(d.id) for d in stale_docs]
    logger.info(
        "Deleting stale portfolio docs",
        count=len(stale_ids),
        sources=[d.source for d in stale_docs],
    )

    qdrant = get_qdrant_client()
    try:
        await qdrant.client.delete(
            collection_name=settings.qdrant_personal_collection,
            points_selector=qdrant_models.FilterSelector(
                filter=qdrant_models.Filter(
                    must=[qdrant_models.FieldCondition(
                        key="document_id",
                        match=qdrant_models.MatchAny(any=stale_ids),
                    )]
                )
            ),
        )
    except Exception as e:
        logger.warning(f"Failed to batch-delete stale Qdrant points", error=str(e))

    for d in stale_docs:
        await session.delete(d)
    await session.flush()


# ─── Per-document ingestion ─────────────────────────────────────────────────

async def _ingest_portfolio_doc(
    raw_markdown: str,
    source_id: str,
    title: str,
    metadata: dict,
    session: AsyncSession,
    force_reingest: bool = False,
) -> int:
    """Embed + upsert one document into Personal_Portfolio. Returns chunk count
    (0 if skipped because content unchanged)."""
    content_hash = hashlib.sha256(raw_markdown.encode()).hexdigest()

    result = await session.execute(select(Document).where(Document.source == source_id))
    existing: Document | None = result.scalars().first()

    if existing and existing.content_hash == content_hash and not force_reingest:
        logger.info("Skipping unchanged portfolio doc", source=source_id)
        return 0

    document_id = str(existing.id) if existing else str(uuid.uuid4())

    # DEFENSIVE: nuke any pre-existing Qdrant chunks for this source BEFORE
    # inserting, regardless of PG Document state. This prevents accumulation
    # when Document rows and Qdrant points get out of sync (e.g., after a
    # crash, manual delete, or a different test session). Cheap — single API
    # call filtered by indexed `source` field.
    qdrant_pre = get_qdrant_client()
    try:
        await qdrant_pre.client.delete(
            collection_name=settings.qdrant_personal_collection,
            points_selector=qdrant_models.FilterSelector(
                filter=qdrant_models.Filter(
                    must=[qdrant_models.FieldCondition(
                        key="source",
                        match=qdrant_models.MatchValue(value=source_id),
                    )]
                )
            ),
        )
    except Exception as e:
        logger.warning("Defensive delete-by-source failed (continuing)", source=source_id, error=str(e))

    if existing:
        # Wipe stale chunks
        qdrant = get_qdrant_client()
        try:
            await qdrant.client.delete(
                collection_name=settings.qdrant_personal_collection,
                points_selector=qdrant_models.FilterSelector(
                    filter=qdrant_models.Filter(
                        must=[qdrant_models.FieldCondition(
                            key="document_id",
                            match=qdrant_models.MatchValue(value=document_id),
                        )]
                    )
                ),
            )
        except Exception as e:
            logger.warning("Failed to delete stale Qdrant points", error=str(e))

        await session.execute(delete(Chunk).where(Chunk.document_id == document_id))
        await session.flush()

        existing.ingestion_state = "processing"
        existing.content_hash = content_hash
        existing.metadata_ = metadata
        existing.title = title
    else:
        doc = Document(
            id=document_id,
            title=title,
            source=source_id,
            content_hash=content_hash,
            metadata_=metadata,
            ingestion_state="processing",
        )
        session.add(doc)

    await session.flush()

    # Build LlamaDocument + chunk
    ensure_llamaindex_configured(chunk_size=512, chunk_overlap=50)
    llama_doc = LlamaDocument(
        text=raw_markdown,
        doc_id=document_id,
        metadata={"document_id": document_id, "title": title, **metadata},
    )
    # Overview, profile, and knowledge docs must stay as a SINGLE chunk so
    # retrieval surfaces the complete content together.
    is_single_chunk = metadata.get("doc_type") in ("overview", "profile", "knowledge")

    if is_single_chunk:
        # Bypass MarkdownNodeParser entirely — header-splitting would split
        # multi-section knowledge files (H2/H3) into many chunks. Single-chunk
        # doc types are kept whole regardless of internal headings.
        from llama_index.core.schema import TextNode
        header_nodes: list[BaseNode] = [
            TextNode(text=raw_markdown, metadata=dict(llama_doc.metadata or {}))
        ]
    else:
        parser = MarkdownNodeParser()
        nodes = parser.get_nodes_from_documents([llama_doc])

    nodes = [n for n in nodes if n.text and n.text.strip()]  # type: ignore[attr-defined]  # TextNode at runtime
    if not nodes:
        logger.warning("No nodes produced for portfolio doc", source=source_id)
        return 0

    for node in nodes:
        node.metadata.update(metadata)

    qdrant = get_qdrant_client()
    await qdrant.ensure_personal_collection()

    vector_store = qdrant.get_vector_store(settings.qdrant_personal_collection, enable_hybrid=True)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    try:
        index = VectorStoreIndex(nodes=[], storage_context=storage_context, show_progress=False)
        await index.ainsert_nodes(nodes)
    except Exception as e:
        logger.error("Failed to insert portfolio nodes into Qdrant", error=str(e))
        raise

    total_tokens = 0
    for i, node in enumerate(nodes):
        tokens = count_tokens(node.text)  # type: ignore[attr-defined]  # TextNode at runtime
        total_tokens += tokens
        session.add(
            Chunk(
                id=node.node_id,
                document_id=document_id,
                chunk_index=i,
                text=node.text,  # type: ignore[attr-defined]  # TextNode at runtime
                token_count=tokens,
                qdrant_point_id=node.node_id,
                metadata_={**metadata, "header_path": node.metadata.get("Header_1", "")},
            )
        )

    if existing:
        existing.ingestion_state = "completed"
        existing.total_chunks = len(nodes)
    else:
        doc.ingestion_state = "completed"
        doc.total_chunks = len(nodes)

    await session.flush()
    logger.info("Portfolio doc ingested", source=source_id, chunks=len(nodes), tokens=total_tokens)
    return len(nodes)


# ─── Main entrypoint ────────────────────────────────────────────────────────

async def refresh_profile_only(session: AsyncSession) -> dict[str, Any]:
    """Re-ingest only `data/personal/profile.md` and flush the askfer cache.

    Cheap path used by the file-watcher in app/worker.py — avoids re-scraping
    the website when the user only edited their bio. Returns a dict with
    "status": "ok" | "skipped" | "error" plus extra context.
    """
    profile = _load_profile_markdown()
    if not profile:
        return {"status": "skipped", "reason": "no_profile_file"}

    md_text, title = profile
    profile_source = "portfolio://profile"
    profile_meta = {
        "doc_type": "profile",
        "scraped_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source": profile_source,
    }

    try:
        chunks = await _ingest_portfolio_doc(
            raw_markdown=md_text,
            source_id=profile_source,
            title=title,
            metadata=profile_meta,
            session=session,
            force_reingest=True,
        )
    except Exception as exc:
        logger.error("refresh_profile_only ingest failed", error=str(exc))
        return {"status": "error", "stage": "ingest", "error": str(exc)}

    try:
        from app.utils.cache import flush_cache_by_namespace
        await flush_cache_by_namespace("askfer")
    except Exception as exc:
        logger.warning("refresh_profile_only cache flush failed", error=str(exc))
        # Cache flush is best-effort — ingest already succeeded.

    return {"status": "ok", "chunks": chunks}


async def sync_portfolio_knowledge_base(
    session: AsyncSession,
    force_reingest: bool = False,
) -> dict[str, Any]:
    """
    Full sync of portfolio website (homepage + projects) and CV PDF into the
    Personal_Portfolio Qdrant collection.
    """
    logger.info("Starting portfolio sync", force_reingest=force_reingest)
    summary: dict[str, Any] = {
        "docs_processed": 0,
        "chunks_ingested": 0,
        "docs_skipped": 0,
        "errors": [],
    }
    scraped_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    # expected_sources is built PROGRESSIVELY as sources are DISCOVERED
    # (not as ingestions succeed — that was the previous `current_sources`
    # bug: a mid-sync failure left the list incomplete and the stale
    # cleanup at the end destroyed live KB data). Sources are added the
    # moment we know we're going to ingest them, regardless of whether
    # the ingest itself succeeds.
    expected_sources: list[str] = []
    project_payloads: list[dict] = []  # Collected for overview doc generation

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Sitemap is needed only when we have to fall back to scraping. We
        # fetch it once up front; on Proxmox / offline, this can fail and we
        # rely entirely on local md files (the common steady-state case).
        try:
            urls = await _fetch_sitemap_urls(client)
        except Exception as exc:
            logger.warning(f"Sitemap unavailable, will fall back to local files: {exc}")
            urls = {
                "homepage": settings.portfolio_homepage_url,
                "projects": [],
                "cv": settings.portfolio_cv_url,
            }

        # After URL discovery: the three sources we always expect (homepage,
        # CV, profile). Profile is always "portfolio://profile" — see step 4.
        expected_sources.extend([
            urls["homepage"],
            urls["cv"],
            "portfolio://profile",
        ])

        # 1. Homepage — local-first
        try:
            local = _load_local_with_title(LOCAL_HOMEPAGE_PATH, "Ferdy Fadhil Lazuardi — Portfolio")
            if local:
                md_text, title = local
                logger.info("Homepage sourced from local file", path=LOCAL_HOMEPAGE_PATH)
            else:
                md_text, title = await _scrape_web_page(client, urls["homepage"])
                md_text = _patch_homepage_counters(md_text)
                _save_local_with_title(LOCAL_HOMEPAGE_PATH, md_text, title or "Ferdy Fadhil Lazuardi — Portfolio")
                logger.info("Homepage scraped + saved locally", path=LOCAL_HOMEPAGE_PATH)

            meta = _build_metadata("homepage", urls["homepage"], scraped_at)
            chunks = await _ingest_portfolio_doc(
                raw_markdown=md_text,
                source_id=urls["homepage"],
                title=title or "Ferdy Fadhil Lazuardi — Portfolio",
                metadata=meta,
                session=session,
                force_reingest=force_reingest,
            )
            if chunks == 0:
                summary["docs_skipped"] += 1
            else:
                summary["docs_processed"] += 1
                summary["chunks_ingested"] += chunks
        except Exception as exc:
            logger.error("Homepage ingest failed", error=str(exc))
            summary["errors"].append(f"homepage: {exc}")

        # 2. Project pages — local-first per slug.
        # If LOCAL_PROJECTS_DIR has any *.md files, use ALL of them (and ignore
        # sitemap). Otherwise, scrape each URL from sitemap and persist locally.
        import os
        local_project_files: list[str] = []
        if os.path.isdir(LOCAL_PROJECTS_DIR):
            local_project_files = sorted(
                f for f in os.listdir(LOCAL_PROJECTS_DIR) if f.endswith(".md")
            )

        if local_project_files:
            logger.info(
                "Project pages sourced from local files",
                dir=LOCAL_PROJECTS_DIR, count=len(local_project_files),
            )
            # Add all local project sources to expected_sources up front,
            # BEFORE any ingest call. If a project ingest fails mid-sync,
            # the old project data is still in the expected set and the
            # stale cleanup preserves it (not deletes it).
            expected_sources.extend(
                settings.portfolio_homepage_url.rstrip("/") + f"/projects/{fname[:-3]}/"
                for fname in local_project_files
            )
            for fname in local_project_files:
                slug = fname[:-3]  # strip .md
                project_url = settings.portfolio_homepage_url.rstrip("/") + f"/projects/{slug}/"
                local_path = os.path.join(LOCAL_PROJECTS_DIR, fname)
                local = _load_local_with_title(local_path, slug)
                if not local:
                    continue
                md_text, title = local
                try:
                    meta = _build_metadata("project", project_url, scraped_at, project_slug=slug)
                    chunks = await _ingest_portfolio_doc(
                        raw_markdown=md_text,
                        source_id=project_url,
                        title=title or slug,
                        metadata=meta,
                        session=session,
                        force_reingest=force_reingest,
                    )
                    if chunks == 0:
                        summary["docs_skipped"] += 1
                    else:
                        summary["docs_processed"] += 1
                        summary["chunks_ingested"] += chunks
                    project_payloads.append({"markdown": md_text, "url": project_url, "title": title or slug})
                except Exception as exc:
                    logger.error("Local project ingest failed", file=fname, error=str(exc))
                    summary["errors"].append(f"projects/{fname}: {exc}")
        else:
            # Sitemap path: extend expected_sources with the full list
            # before iterating. Same reasoning as local path above.
            expected_sources.extend(urls["projects"])
            for project_url in urls["projects"]:
                try:
                    md_text, title = await _scrape_web_page(client, project_url)
                    slug = _slugify_project_url(project_url)
                    _save_local_with_title(
                        os.path.join(LOCAL_PROJECTS_DIR, f"{slug}.md"),
                        md_text,
                        title or slug,
                    )
                    meta = _build_metadata("project", project_url, scraped_at, project_slug=slug)
                    chunks = await _ingest_portfolio_doc(
                        raw_markdown=md_text,
                        source_id=project_url,
                        title=title or slug,
                        metadata=meta,
                        session=session,
                        force_reingest=force_reingest,
                    )
                    if chunks == 0:
                        summary["docs_skipped"] += 1
                    else:
                        summary["docs_processed"] += 1
                        summary["chunks_ingested"] += chunks
                    project_payloads.append({"markdown": md_text, "url": project_url, "title": title or slug})
                except Exception as exc:
                    logger.error("Project scrape failed", url=project_url, error=str(exc))
                    summary["errors"].append(f"{project_url}: {exc}")

        # 3. CV — local-first
        try:
            local = _load_local_with_title(LOCAL_CV_PATH, "Curriculum Vitae")
            if local:
                md_text, title = local
                logger.info("CV sourced from local file", path=LOCAL_CV_PATH)
            else:
                md_text, title = await _scrape_cv_pdf(client, urls["cv"])
                _save_local_with_title(LOCAL_CV_PATH, md_text, title or "Curriculum Vitae")
                logger.info("CV scraped + saved locally", path=LOCAL_CV_PATH)

            meta = _build_metadata("cv", urls["cv"], scraped_at)
            chunks = await _ingest_portfolio_doc(
                raw_markdown=md_text,
                source_id=urls["cv"],
                title=title,
                metadata=meta,
                session=session,
                force_reingest=force_reingest,
            )
            if chunks == 0:
                summary["docs_skipped"] += 1
            else:
                summary["docs_processed"] += 1
                summary["chunks_ingested"] += chunks
        except Exception as exc:
            logger.error("CV ingest failed", error=str(exc))
            summary["errors"].append(f"cv: {exc}")

        # 4. Editable profile (data/personal/profile.md). Single source of truth
        # for "who is Ferdy" answers — separate from CV PDF and homepage so it's
        # easy to edit without re-scraping the website.
        try:
            profile = _load_profile_markdown()
            if profile:
                md_text, title = profile
                profile_source = "portfolio://profile"
                profile_meta = {
                    "doc_type": "profile",
                    "scraped_at": scraped_at,
                    "source": profile_source,
                }
                profile_chunks = await _ingest_portfolio_doc(
                    raw_markdown=md_text,
                    source_id=profile_source,
                    title=title,
                    metadata=profile_meta,
                    session=session,
                    force_reingest=True,  # always rebuild to reflect latest edits
                )
                if profile_chunks > 0:
                    summary["docs_processed"] += 1
                    summary["chunks_ingested"] += profile_chunks
        except Exception as exc:
            logger.error("Profile doc ingestion failed", error=str(exc))
            summary["errors"].append(f"profile: {exc}")

        # 4b. Editable knowledge files (data/personal/knowledge/*.md). One file
        # per topic — methodologies, frameworks, opinions Askfer should be able
        # to answer about (e.g. "How do you use Bloom's Taxonomy?", "How do you
        # apply ADDIE?"). Each file is ingested as a single chunk.
        try:
            if os.path.isdir(LOCAL_KNOWLEDGE_DIR):
                knowledge_files = sorted(
                    f for f in os.listdir(LOCAL_KNOWLEDGE_DIR) if f.endswith(".md")
                )
                # Add all knowledge sources to expected_sources up front,
                # BEFORE any ingest call. Same reasoning as projects.
                expected_sources.extend(
                    f"portfolio://knowledge/{fname[:-3]}"
                    for fname in knowledge_files
                )
                for fname in knowledge_files:
                    slug = fname[:-3]
                    local_path = os.path.join(LOCAL_KNOWLEDGE_DIR, fname)
                    local = _load_local_with_title(local_path, slug.replace("-", " ").title())
                    if not local:
                        continue
                    md_text, title = local
                    knowledge_source = f"portfolio://knowledge/{slug}"
                    knowledge_meta = {
                        "doc_type": "knowledge",
                        "knowledge_slug": slug,
                        "scraped_at": scraped_at,
                        "source": knowledge_source,
                    }
                    try:
                        k_chunks = await _ingest_portfolio_doc(
                            raw_markdown=md_text,
                            source_id=knowledge_source,
                            title=title or slug,
                            metadata=knowledge_meta,
                            session=session,
                            force_reingest=True,  # rebuild on every sync
                        )
                        if k_chunks > 0:
                            summary["docs_processed"] += 1
                            summary["chunks_ingested"] += k_chunks
                    except Exception as exc:
                        logger.error("Knowledge doc ingest failed", file=fname, error=str(exc))
                        summary["errors"].append(f"knowledge/{fname}: {exc}")
        except Exception as exc:
            logger.error("Knowledge dir scan failed", error=str(exc))
            summary["errors"].append(f"knowledge: {exc}")

        # 5. Build & ingest the OVERVIEW document
        # This single doc lists all projects grouped by org_label so a query like
        # "what projects have you done?" returns a complete index instead of just
        # the top-K retrieved chunks.
        if project_payloads:
            try:
                summaries = []
                for p in project_payloads:
                    try:
                        s_item = await _summarize_project_for_overview(
                            p["markdown"], p["url"], p["title"]
                        )
                        summaries.append(s_item)
                    except Exception as exc:
                        logger.warning(
                            "Overview summarize failed for project",
                            url=p["url"], error=str(exc),
                        )

                overview_md = _build_overview_markdown(summaries)
                if overview_md:
                    overview_source = "portfolio://overview"
                    # Add to expected_sources BEFORE ingest: if the overview
                    # ingest fails, the previous overview doc is preserved
                    # rather than deleted by the stale cleanup.
                    expected_sources.append(overview_source)
                    overview_meta = {
                        "doc_type": "overview",
                        "scraped_at": scraped_at,
                        "source": overview_source,
                    }
                    overview_chunks = await _ingest_portfolio_doc(
                        raw_markdown=overview_md,
                        source_id=overview_source,
                        title="Portfolio Overview — All Projects",
                        metadata=overview_meta,
                        session=session,
                        force_reingest=True,  # always rebuild to reflect latest scrape
                    )
                    if overview_chunks > 0:
                        summary["docs_processed"] += 1
                        summary["chunks_ingested"] += overview_chunks
            except Exception as exc:
                logger.error("Overview doc build failed", error=str(exc))
                summary["errors"].append(f"overview: {exc}")

        # 6. Stale cleanup (only if we got at least one source — avoids wiping
        # everything during a total network outage). Uses expected_sources
        # (the complete discovery-time set) rather than an incrementally-
        # built list of "sources that successfully ingested" — see the
        # comment on the variable declaration above.
        if expected_sources:
            try:
                await _delete_stale_portfolio_docs(session, expected_sources)
            except Exception as exc:
                logger.warning("Stale cleanup failed", error=str(exc))
                summary["errors"].append(f"cleanup: {exc}")

    logger.info("Portfolio sync complete", **{k: v for k, v in summary.items() if k != "errors"})
    return summary


# ─── CLI entry point ─────────────────────────────────────────────────────────
# Run from container: `uv run python -m app.ingestion.portfolio_sync`
# Or with force flag : `uv run python -m app.ingestion.portfolio_sync --force`

async def _cli_main(force_reingest: bool) -> None:
    from app.database.postgres import AsyncSessionLocal
    print(f"Starting portfolio sync (force_reingest={force_reingest})...")
    async with AsyncSessionLocal() as session:
        result = await sync_portfolio_knowledge_base(session, force_reingest=force_reingest)
        await session.commit()
    print("=" * 50)
    print("SYNC RESULT:")
    for k, v in result.items():
        print(f"  {k}: {v}")
    print("=" * 50)


if __name__ == "__main__":
    import sys
    force = "--force" in sys.argv or "-f" in sys.argv
    import asyncio
    asyncio.run(_cli_main(force_reingest=force))
