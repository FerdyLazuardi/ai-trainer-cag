"""Self-check for conversation_state.bump_topic_streak streak/decay logic.

Run: python -m app.tests.test_topic_streak  (asserts, no framework)
"""
import asyncio

from app.agents import conversation_state as cs


class _FakePipe:
    def __init__(self, store):
        self.store = store

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    def hsetex(self, key, mapping=None, ex=None):
        self.store.setdefault(key, {}).update(mapping or {})

    def expire(self, key, ttl):
        pass

    async def execute(self):
        pass


class _FakeRedis:
    def __init__(self):
        self.store = {}

    async def hget(self, key, field):
        return self.store.get(key, {}).get(field)

    def pipeline(self, transaction=True):
        return _FakePipe(self.store)


async def _main():
    r = _FakeRedis()
    conv = "c1"

    # Same topic 3x with threshold 3 → fires on the 3rd, then resets.
    assert await cs.bump_topic_streak(r, conv, "A", threshold=3) is None  # n=1
    assert await cs.bump_topic_streak(r, conv, "A", threshold=3) is None  # n=2
    assert await cs.bump_topic_streak(r, conv, "A", threshold=3) == "A"   # n=3 → fire
    # Reset after firing — next same-topic turn starts the count over.
    assert await cs.bump_topic_streak(r, conv, "A", threshold=3) is None  # n=1

    # Decay: a single topic shift drops count by 1, doesn't hard-reset.
    r2 = _FakeRedis()
    await cs.bump_topic_streak(r2, "c2", "A", threshold=3)               # A n=1
    await cs.bump_topic_streak(r2, "c2", "A", threshold=3)               # A n=2
    await cs.bump_topic_streak(r2, "c2", "B", threshold=3)               # shift → A n=1 (decayed, topic kept)
    assert await cs.bump_topic_streak(r2, "c2", "A", threshold=3) is None  # A n=2 (back on topic)
    assert await cs.bump_topic_streak(r2, "c2", "A", threshold=3) == "A"   # A n=3 → fire

    # Empty/blank topic → never fires, never throws.
    assert await cs.bump_topic_streak(r, conv, "", threshold=3) is None
    assert await cs.bump_topic_streak(r, "", "A", threshold=3) is None

    print("test_topic_streak OK")


if __name__ == "__main__":
    asyncio.run(_main())
