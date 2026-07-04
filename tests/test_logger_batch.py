import asyncio

import pytest


@pytest.mark.asyncio
async def test_batch_logger_stop_waits_for_pending_insert(monkeypatch):
    from app.utils import logger_batch

    gate = asyncio.Event()
    inserted = []

    async def fake_insert(row):
        await gate.wait()
        inserted.append(row)

    monkeypatch.setattr(logger_batch, "_do_insert", fake_insert)

    batch = logger_batch.BatchLogger()
    await batch.start()
    await batch.add_log({"query": "hello"})
    await asyncio.sleep(0)

    stop_task = asyncio.create_task(batch.stop())
    await asyncio.sleep(0)
    try:
        assert not stop_task.done()
    finally:
        gate.set()

    await stop_task
    assert inserted and inserted[0]["query"] == "hello"

