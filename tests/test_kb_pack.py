from app.knowledge.kb_pack import assemble_kb_pack
from app.knowledge.moodle_markdown import MoodleMarkdownFile


def test_assemble_kb_pack_is_stable_and_excludes_generated_timestamp():
    docs = [
        MoodleMarkdownFile(
            course_id=2,
            section_id=2,
            section_name="B & B",
            filename="b.md",
            content="Line\r\nTwo",
            content_hash="ignored",
        ),
        MoodleMarkdownFile(
            course_id=1,
            section_id=1,
            section_name="A",
            filename="a.md",
            content="# A",
            content_hash="ignored",
        ),
    ]

    pack_a = assemble_kb_pack(docs)
    pack_b = assemble_kb_pack(list(reversed(docs)))

    assert pack_a == pack_b
    assert pack_a.kb_hash
    assert 'generated_at="' not in pack_a.text
    assert pack_a.text.startswith(f'<knowledge_base version="sha256:{pack_a.kb_hash}">\n')
    assert "- [DOC-001] Course: 1\n  Section: A\n  File: a.md" in pack_a.text
    assert '<doc id="DOC-002" course="2" section="B &amp; B" file="b.md" roles="ALL">\nLine\nTwo\n</doc>' in pack_a.text


def test_assemble_kb_pack_parses_roles():
    docs = [
        MoodleMarkdownFile(
            course_id=1,
            section_id=1,
            section_name="A",
            filename="a.md",
            content="<!-- roles: BP, BM -->\n# Title",
            content_hash="ignored",
        ),
        MoodleMarkdownFile(
            course_id=2,
            section_id=2,
            section_name="B",
            filename="b.md",
            content="---\ncourse_name: Amartha\nroles: [HO]\n---\n# Title",
            content_hash="ignored",
        ),
    ]

    pack = assemble_kb_pack(docs)
    assert 'roles="BP,BM"' in pack.text
    assert 'roles="HO"' in pack.text


def test_extract_sections_from_kb_pack_text():
    from app.knowledge.kb_pack import extract_kb_sections, extract_kb_topics

    kb_text = "\n".join([
        '<knowledge_base version="sha256:x">',
        '<doc id="DOC-001" course="3" section="Business Process" file="Validasi UK.md">',
        "A",
        "</doc>",
        '<doc id="DOC-002" course="3" section="Business Process" file="Pelayanan.md">',
        "B",
        "</doc>",
        '<doc id="DOC-003" course="3" section="Product &amp; Policy" file="Modal.md">',
        "C",
        "</doc>",
        "</knowledge_base>",
    ])

    assert extract_kb_topics(kb_text) == ["Business Process", "Product & Policy"]
    assert extract_kb_sections(kb_text) == {
        "Business Process": ["Validasi UK", "Pelayanan"],
        "Product & Policy": ["Modal"],
    }
