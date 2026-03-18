import sys
import os
import asyncio

sys.path.append('/app')

from app.llm.client import get_llm
from langchain_core.messages import SystemMessage, HumanMessage

async def test_llm():
    print("Testing Direct LLM Call...")
    try:
        llm = get_llm()
        print(f"✅ LLM Client created: {llm.model_name}")
        
        response = await llm.ainvoke([
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="Hello, respond with 'OK' if you see this.")
        ])
        print(f"✅ LLM Response: {response.content}")
        
    except Exception as e:
        print(f"❌ Error during LLM test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_llm())
