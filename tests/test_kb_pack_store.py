import pytest


class FakeScalars:
    def __init__(self, item):
        self.item = item

    def first(self):
        return self.item


class FakeResult:
    def __init__(self, item=None):
        self.item = item

    def scalars(self):
        return FakeScalars(self.item)


class FakeSession:
    def __init__(self, active=None):
        self.active = active
        self.executed = []
        self.added = []
        self.flushed = False

    async def execute(self, stmt):
        self.executed.append(stmt)
        return FakeResult(self.active if len(self.executed) == 1 else None)

    def add(self, obj):
        self.added.append(obj)

    async def flush(self):
        self.flushed = True


@pytest.mark.asyncio
async def test_save_active_kb_pack_noops_when_hash_unchanged():
    from app.database.models import KnowledgeBasePack
    from app.knowledge.store import save_active_kb_pack

    active = KnowledgeBasePack(source="moodle", kb_hash="same", content="old", token_count=1, is_active=True)
    session = FakeSession(active=active)

    result = await save_active_kb_pack(session, source="moodle", kb_hash="same", content="new", token_count=2)

    assert result is active
    assert session.added == []
    assert session.flushed is False
    assert len(session.executed) == 1


@pytest.mark.asyncio
async def test_save_active_kb_pack_deactivates_old_and_adds_changed_pack():
    from app.database.models import KnowledgeBasePack
    from app.knowledge.store import save_active_kb_pack

    active = KnowledgeBasePack(source="moodle", kb_hash="old", content="old", token_count=1, is_active=True)
    session = FakeSession(active=active)

    result = await save_active_kb_pack(session, source="moodle", kb_hash="new", content="content", token_count=3)

    assert result is session.added[0]
    assert result.source == "moodle"
    assert result.kb_hash == "new"
    assert result.content == "content"
    assert result.token_count == 3
    assert result.is_active is True
    assert session.flushed is True
    assert len(session.executed) == 2


@pytest.mark.asyncio
async def test_sync_moodle_kb_pack_flushes_answer_cache_on_hash_change(monkeypatch):
    from app.knowledge import sync
    from app.knowledge.moodle_markdown import MoodleMarkdownFile

    flushed = False

    async def fake_pull(*args, **kwargs):
        return [
            MoodleMarkdownFile(
                course_id=3,
                section_id=1,
                section_name="A",
                filename="a.md",
                content="new content",
                content_hash="ignored",
            )
        ]

    async def fake_get_active(session, source):
        class Active:
            kb_hash = "old"

        return Active()

    async def fake_save_active_kb_pack(session, **kwargs):
        class Saved:
            id = 1
            kb_hash = kwargs["kb_hash"]
            token_count = 2

        return Saved()

    async def fake_flush_cache():
        nonlocal flushed
        flushed = True

    monkeypatch.setattr(sync, "pull_moodle_markdown", fake_pull)
    monkeypatch.setattr(sync, "get_active_kb_pack", fake_get_active)
    monkeypatch.setattr(sync, "save_active_kb_pack", fake_save_active_kb_pack)
    monkeypatch.setattr(sync, "flush_cache", fake_flush_cache)

    result = await sync.sync_moodle_kb_pack(session=object(), course_id=3)

    assert result["status"] == "updated"
    assert flushed is True


@pytest.mark.asyncio
async def test_sync_moodle_kb_pack_keeps_cache_when_hash_unchanged(monkeypatch):
    from app.knowledge import sync
    from app.knowledge.kb_pack import assemble_kb_pack
    from app.knowledge.moodle_markdown import MoodleMarkdownFile

    docs = [
        MoodleMarkdownFile(
            course_id=3,
            section_id=1,
            section_name="A",
            filename="a.md",
            content="same content",
            content_hash="ignored",
        )
    ]
    same_hash = assemble_kb_pack(docs).kb_hash
    flushed = False

    async def fake_pull(*args, **kwargs):
        return docs

    async def fake_get_active(session, source):
        class Active:
            kb_hash = same_hash

        return Active()

    async def fake_save_active_kb_pack(session, **kwargs):
        class Saved:
            id = 1
            kb_hash = same_hash
            token_count = 2

        return Saved()

    async def fake_flush_cache():
        nonlocal flushed
        flushed = True

    monkeypatch.setattr(sync, "pull_moodle_markdown", fake_pull)
    monkeypatch.setattr(sync, "get_active_kb_pack", fake_get_active)
    monkeypatch.setattr(sync, "save_active_kb_pack", fake_save_active_kb_pack)
    monkeypatch.setattr(sync, "flush_cache", fake_flush_cache)

    result = await sync.sync_moodle_kb_pack(session=object(), course_id=3)

    assert result["status"] == "unchanged"
    assert flushed is False


def test_clear_cag_kb_cache_resets_metadata_caches():
    from app.graph import pipeline

    pipeline._active_kb_cache.update({"hash": "x", "content": "KB", "expires_at": 9999999999.0})
    pipeline._course_cache.update({"courses": ["A"], "expires_at": 9999999999.0})
    pipeline._section_map_cache.update({"map": {"A": ["a"]}, "expires_at": 9999999999.0})
    pipeline._h2_topics_cache.update({"map": {"ALL": ["Topic H2"]}, "expires_at": 9999999999.0})

    pipeline.clear_cag_kb_cache()

    assert pipeline._active_kb_cache["content"] == ""
    assert pipeline._course_cache["courses"] == []
    assert pipeline._section_map_cache["map"] == {}
    assert pipeline._h2_topics_cache["map"] == {}


@pytest.mark.asyncio
async def test_extract_and_load_h2_topics_by_role(monkeypatch):
    from app.graph import pipeline
    from app.knowledge.kb_pack import extract_kb_h2_headings

    sample_kb = (
        '<knowledge_base version="sha256:test">\n'
        '<doc id="DOC-001" course="3" section="SOP" file="sop.md" roles="BP">\n'
        '# Judul Utama\n'
        '## **Prosedur Pencairan Pinjaman**\n'
        '## Catatan\n'
        '</doc>\n'
        '<doc id="DOC-002" course="3" section="Leadership" file="bm.md" roles="BM">\n'
        '## Evaluasi Kinerja Cabang\n'
        '</doc>\n'
        '</knowledge_base>'
    )

    assert extract_kb_h2_headings(sample_kb) == [
        "Prosedur Pencairan Pinjaman",
        "Evaluasi Kinerja Cabang",
    ]

    async def fake_load_kb():
        return sample_kb

    pipeline.clear_cag_kb_cache()
    monkeypatch.setattr(pipeline, "_load_active_cag_kb_text", fake_load_kb)

    bp_h2 = await pipeline._load_h2_topics("BP")
    bm_h2 = await pipeline._load_h2_topics("BM")

    assert bp_h2 == ["Prosedur Pencairan Pinjaman"]
    assert bm_h2 == ["Evaluasi Kinerja Cabang"]
    pipeline.clear_cag_kb_cache()

