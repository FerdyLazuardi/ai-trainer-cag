from pathlib import Path


def test_chat_markdown_rendering_goes_through_sanitizer():
    script = Path("app/static/script.js").read_text(encoding="utf-8")

    assert "function renderMarkdownSafe" in script
    assert "innerHTML = marked.parse" not in script
