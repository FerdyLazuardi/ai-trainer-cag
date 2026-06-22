import asyncio
from app.llm.client import get_llm, get_cheap_llm, get_chat_llm, get_empathy_llm
from app.utils.logger_batch import batch_logger
from app.database.redis_client import get_redis_client

async def main():
    print("--- 1. Testing LLM Factories ---")
    try:
        main_llm = get_llm()
        print(f"Main LLM: model={main_llm.model_name}, temp={main_llm.temperature}, retries={main_llm.max_retries}")
        
        cheap_llm = get_cheap_llm()
        print(f"Cheap LLM: model={cheap_llm.model_name}, temp={cheap_llm.temperature}, retries={cheap_llm.max_retries}")
        
        chat_llm = get_chat_llm()
        print(f"Chat LLM: model={chat_llm.model_name}, temp={chat_llm.temperature}, retries={chat_llm.max_retries}")
        
        empathy_llm = get_empathy_llm()
        print(f"Empathy LLM: model={empathy_llm.model_name}, temp={empathy_llm.temperature}, retries={empathy_llm.max_retries}")
    except Exception as e:
        print(f"ERROR in LLM instantiation: {e}")

    print("\n--- 2. Testing Redis Client Instantiation ---")
    try:
        redis_client = get_redis_client()
        print(f"Redis Client instantiated successfully: {redis_client}")
    except Exception as e:
        print(f"ERROR in Redis Client: {e}")

    print("\n--- 3. Testing Batch Logger ---")
    try:
        # Fire and forget
        await batch_logger.add_log({
            "query": "Test query from ponytail audit verification",
            "answer": "All systems go.",
            "session_id": "ponytail_test_1",
            "user_id": 999
        })
        print("Log entry dispatched to DB successfully. Waiting 1 second for background task to execute...")
        await asyncio.sleep(1)
        print("Done!")
    except Exception as e:
        print(f"ERROR in Batch Logger: {e}")

if __name__ == "__main__":
    asyncio.run(main())
