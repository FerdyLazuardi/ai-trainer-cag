"""Debug script: inspect raw intent_scores from one eval case."""
import asyncio
from app.eval.runner import run_one_query


async def main():
    cases = [
        ("mentor_06 should have L=0.8-1.0", "ajarin aku de-escalation, masih bingung"),
        ("mentor_01 should have L=0.7-1.0", "ajarin aku cara handle Mitra yang marah"),
        ("mentor_03 should have L=0.0-0.3", "lapor pelecehan gimana?"),
    ]
    for label, q in cases:
        r = await run_one_query(query=q)
        print(f"\n=== {label} ===")
        print(f"  query:   {q}")
        print(f"  intent:  {r.get('intent')}")
        print(f"  intent_scores: {r.get('intent_scores')}")
        print(f"  type(intent_scores): {type(r.get('intent_scores'))}")
        is_ = r.get("intent_scores") or {}
        print(f"  learning_context value: {is_.get('learning_context')!r}  type: {type(is_.get('learning_context'))}")


if __name__ == "__main__":
    asyncio.run(main())
