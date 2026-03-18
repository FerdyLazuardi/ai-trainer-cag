import sys
import asyncio
sys.path.append('/app')

from app.graph.pipeline import get_rag_graph
from langchain_core.messages import HumanMessage

async def test():
    print("=== Test 1: Greeting ===")
    try:
        graph = get_rag_graph()
        print("✅ Graph compiled")
        result = await graph.ainvoke({
            "messages": [HumanMessage(content="hi")],
        })
        print(f"✅ Response: {result['messages'][-1].content[:100]}")
        print(f"   Intent: {result.get('intent', 'N/A')}")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

    print("\n=== Test 2: Knowledge ===")
    try:
        graph = get_rag_graph()
        result = await graph.ainvoke({
            "messages": [HumanMessage(content="apa itu client protection?")],
        })
        print(f"✅ Response: {result['messages'][-1].content[:200]}")
        print(f"   Intent: {result.get('intent', 'N/A')}")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test())
