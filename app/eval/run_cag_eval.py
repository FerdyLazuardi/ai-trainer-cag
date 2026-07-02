import asyncio
import json
import time
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage
from loguru import logger

from app.eval.judge import judge_faithfulness
from app.graph.pipeline import get_rag_graph


DEEPSEEK_V4_FLASH_PRICING = {
    "input_uncached": 0.14 / 1_000_000,
    "input_cached": 0.014 / 1_000_000,
    "output": 0.28 / 1_000_000,
}


async def evaluate_cag_query(query: str, expected_intent: str, min_faithfulness: float) -> dict[str, Any]:
    rag_graph = get_rag_graph()
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "conversation_id": "cag-eval",
        "conversation_summary": "",
        "user_profile": {"summary": "", "course_names": []},
        "user_preferences": None,
    }

    start = time.perf_counter()
    result = await rag_graph.ainvoke(initial_state)
    latency = time.perf_counter() - start

    final_msg = result["messages"][-1]
    answer = final_msg.content if hasattr(final_msg, "content") else str(final_msg)
    intent = result.get("intent")
    
    # Extract token usage from response_metadata
    prompt_tokens = 0
    completion_tokens = 0
    cached_tokens = 0
    
    if hasattr(final_msg, "response_metadata"):
        usage = final_msg.response_metadata.get("token_usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        
        # DeepSeek / OpenRouter cached tokens
        prompt_details = usage.get("prompt_tokens_details") or {}
        cached_tokens = prompt_details.get("cached_tokens", 0)

    # Cost calculation
    uncached_input = max(0, prompt_tokens - cached_tokens)
    cost = (
        (uncached_input * DEEPSEEK_V4_FLASH_PRICING["input_uncached"]) +
        (cached_tokens * DEEPSEEK_V4_FLASH_PRICING["input_cached"]) +
        (completion_tokens * DEEPSEEK_V4_FLASH_PRICING["output"])
    )

    # Faithfulness
    skip_judge = intent in {"GREETING", "AMBIGUOUS", "MALICIOUS", "TOPIC_LIST", "OFF_SCOPE"}
    faithfulness_score = None
    if not skip_judge and min_faithfulness > 0:
        judge = await judge_faithfulness(
            query=query,
            answer=answer,
            retrieved_context=[{"text": result.get("cag_kb_text", "")}] if result.get("cag_kb_text") else [],
        )
        if judge:
            faithfulness_score = judge.score

    return {
        "query": query,
        "intent": intent,
        "intent_match": intent == expected_intent,
        "latency_s": latency,
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "completion_tokens": completion_tokens,
        "cache_ratio": (cached_tokens / prompt_tokens) if prompt_tokens > 0 else 0.0,
        "cost_usd": cost,
        "faithfulness": faithfulness_score,
        "passed_faithfulness": faithfulness_score >= min_faithfulness if faithfulness_score is not None else True,
        "answer_preview": answer[:100].replace("\n", " "),
    }


async def main():
    logger.info("Starting CAG Evaluation")
    
    dataset_path = Path("data/eval/golden_set.jsonl")
    if not dataset_path.exists():
        logger.error(f"Dataset not found: {dataset_path}")
        return

    records = []
    with dataset_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                records.append(json.loads(line))

    logger.info(f"Loaded {len(records)} test cases")

    results = []
    # Using semaphore to avoid rate limits
    sem = asyncio.Semaphore(3)
    
    async def run_with_sem(record):
        async with sem:
            query = record.get("query") or (record.get("turns", [])[-1] if record.get("turns") else "")
            expected_intent = record.get("expected_intent")
            min_faith = float(record.get("min_faithfulness", 0.0))
            return await evaluate_cag_query(query, expected_intent, min_faith)

    tasks = [run_with_sem(r) for r in records[:20]]  # Run first 20 for quick eval
    results = await asyncio.gather(*tasks)

    # Aggregate metrics
    total_latency = sum(r["latency_s"] for r in results)
    total_cost = sum(r["cost_usd"] for r in results)
    avg_cache_ratio = sum(r["cache_ratio"] for r in results) / len(results) if results else 0
    intent_accuracy = sum(1 for r in results if r["intent_match"]) / len(results) if results else 0
    
    faith_results = [r["faithfulness"] for r in results if r["faithfulness"] is not None]
    avg_faithfulness = sum(faith_results) / len(faith_results) if faith_results else 0
    faith_pass_rate = sum(1 for r in results if r["passed_faithfulness"]) / len(results) if results else 0

    print("\n" + "="*40)
    print("🎯 CAG Evaluation Results")
    print("="*40)
    print(f"Total Cases Run      : {len(results)}")
    print(f"Intent Accuracy      : {intent_accuracy:.1%}")
    print(f"Avg Faithfulness     : {avg_faithfulness:.2f}")
    print(f"Faithfulness Pass %  : {faith_pass_rate:.1%}")
    print(f"Avg Latency          : {total_latency / len(results):.2f}s")
    print(f"Avg Cache Hit Ratio  : {avg_cache_ratio:.1%}")
    print(f"Total Eval Cost      : ${total_cost:.5f}")
    print("="*40)


if __name__ == "__main__":
    asyncio.run(main())
