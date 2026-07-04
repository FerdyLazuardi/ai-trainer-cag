import json
import shutil
import subprocess
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


def test_stream_fallback_continues_from_last_received_token():
    if shutil.which("node") is None:
        return

    script = Path("app/static/script.js").read_text(encoding="utf-8")
    fn = _extract_js_function(script, "alignFallbackText")
    streamed = "Ava sudah menjelaskan bagian ini sampai token terakhir"
    displayed = "Ava sudah menjelaskan bagian ini"
    reply = streamed + " lalu lanjut tanpa mengulang dari awal."

    js = f"""
{fn}
const result = alignFallbackText({json.dumps(streamed)}, {json.dumps(displayed)}, {json.dumps(reply)});
console.log(JSON.stringify(result));
"""
    result = subprocess.run(
        ["node", "-e", js],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(result.stdout) == {
        "target": reply,
        "displayed": streamed,
    }


def test_stream_fallback_restarts_renderer_after_alignment():
    script = Path("app/static/script.js").read_text(encoding="utf-8")
    start = script.index("const alignment = alignFallbackText")
    block = script[start : script.index("} else {", start)]

    assert "if (_finalized)" not in block
    assert "_finalized = false;" in block
    assert "smoothStreamWorker();" in block
