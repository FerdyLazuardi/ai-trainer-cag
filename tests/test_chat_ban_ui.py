from pathlib import Path


def test_ban_ui_refresh_delete_and_placeholder_wiring():
    script = Path("app/static/script.js").read_text(encoding="utf-8")
    style = Path("app/static/style.css").read_text(encoding="utf-8")

    assert 'textarea.placeholder = _banActive ? "AI Trainer dinonaktifkan sementara" : "Ketik pesan...";' in script
    assert "#prompt:disabled::placeholder" in style
    assert "font-size: 12px;" in style

    assert "/api/v1/chat/ban-status" in script
    assert "restoreStoredBanCountdown();" in script
    assert "refreshBanStatus();" in script

    clear_start = script.index("async function doClearChat()")
    clear_block = script[clear_start : script.index("// ============================================================", clear_start)]
    assert "const banned = await refreshBanStatus();" in clear_block
    assert "if (!banned) showIntro();" in clear_block

    intro_start = script.index("function showIntro()")
    intro_block = script[intro_start : script.index("async function loadHistory()", intro_start)]
    assert "if (_banActive) return;" in intro_block
