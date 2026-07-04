from pathlib import Path


def _extract_js_function(source: str, name: str) -> str:
    start = source.index(f"function {name}")
    brace = source.index("{", start)
    depth = 0
    for idx in range(brace, len(source)):
        if source[idx] == "{":
            depth += 1
        elif source[idx] == "}":
            depth -= 1
            if depth == 0:
                return source[start : idx + 1]
    raise AssertionError(f"Could not extract {name}")


def test_stream_error_keeps_partial_and_shows_retry_status():
    script = Path("app/static/script.js").read_text(encoding="utf-8")
    error_start = script.index('if (currentEventType === "error")')
    error_block = script[error_start : script.index("if (parsed.token !== undefined)", error_start)]

    assert "showStreamRetryStatus(streamWrap" in error_block
    assert "Connection Failed." in script
    assert 'retryBtn.textContent = "Retry";' in script
    assert "_targetText =" not in error_block


def test_stream_error_does_not_regenerate_non_stream_answer():
    script = Path("app/static/script.js").read_text(encoding="utf-8")
    catch_start = script.index("} catch (err) {")
    catch_block = script[catch_start : script.index("} finally {", catch_start)]

    assert "alignFallbackText" not in script
    assert "Falling back to non-streaming /chat endpoint" not in catch_block
    assert "${baseUrl}/api/v1/chat`" not in catch_block
