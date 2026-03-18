"""
Agentic RAG pipeline assembly with Intent Gating.
"""
from functools import lru_cache
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from loguru import logger

from app.graph.state import RAGState
from app.graph.tools import search_company_knowledge
from app.llm.client import get_llm

# ─── System Prompts ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are Peped, a friendly AI assistant for Amarthapedia (Amartha's LMS). ✨
Tone: Casual and friendly using 'aku/kamu' style. Stay professional when presenting materials! 🔥

MANDATORY RULES:
1. LANGUAGE: Respond in the same language as the user (ID/EN). 🌐
2. RETRIEVAL: For any questions about facts, rules, or materials, you MUST call `search_company_knowledge` immediately.
3. FILTERED PRECISION: Provide the material from the document ACCURATELY but FILTER OUT irrelevant content. For example, if the user asks about Principle 8, focus only on that and remove any mention of other principles (like Principle 5) that might be in the same text. Keep the technical language of the document, but ensure it only answers the user's specific question. 📖
4. LINKS: Only provide a course link if:
   - The user switches to a new topic or course.
   - The user explicitly asks for the link.
   - Format: "Pelajari lebih lanjut di sini: [Course Title](https://semiexpositive-renaldo-unvindictively.ngrok-free.dev/course/view.php?id={course_id})" 🔗
5. If no info is found, politely inform the user in your casual style. No hallucinations! 🚫
"""

ROUTER_PROMPT = """Classify the user intent:
- GREETING: Hello, hi, introduce self, small talk, how are you, who are you?
- AMBIGUOUS: Too short, meaningless words (e.g., "apa", "gimana", "apa itu", "how", "wait"), or vague questions that need clarification.
- KNOWLEDGE: Clear questions about Amartha, products, policies, training, or technical facts.
Respond ONLY with GREETING, AMBIGUOUS, or KNOWLEDGE (one word)."""

# ─── Nodes ───────────────────────────────────────────────────────────────────

def _classify_intent(state: RAGState):
    """Classify user intent before sending to the heavy Agent."""
    user_msg = state["messages"][-1].content
    low_msg = user_msg.lower().strip()
    
    # Fast heuristic for common greetings
    if low_msg in ["halo", "hi", "hey", "pagi", "siang", "sore", "malam", "test", "siapa", "siapa kamu"]:
         return {"intent": "GREETING"}

    llm = get_llm()
    response = llm.invoke([
        SystemMessage(content=ROUTER_PROMPT),
        HumanMessage(content=user_msg)
    ])
    intent = response.content.upper().strip()
    
    # Return one of the three options
    if "GREETING" in intent: return {"intent": "GREETING"}
    if "AMBIGUOUS" in intent: return {"intent": "AMBIGUOUS"}
    return {"intent": "KNOWLEDGE"}


def _handle_ambiguity(state: RAGState):
    """Politely ask for clarification when query is too vague."""
    llm = get_llm()
    ambiguity_sys = "You are Peped, a friendly AI assistant for Amarthapedia. The user provided an ambiguous or too short query like 'apa' or 'siapa' without context. Politely and casually ask them in 'aku/kamu' style to explain more so you can help them better. No tools."
    response = llm.invoke([SystemMessage(content=ambiguity_sys)] + state["messages"])
    return {"messages": [response]}


def _handle_greeting(state: RAGState):
    """Simple friendly response for sapaan/perkenalan."""
    llm = get_llm()
    greet_sys = "You are Peped, AI assistant for Amarthapedia. Greet the user warmly using 'aku/kamu'. Don't use tools."
    response = llm.invoke([SystemMessage(content=greet_sys)] + state["messages"])
    return {"messages": [response]}


def _call_knowledge_agent(state: RAGState):
    """Full Agent flow with search tools."""
    messages = state["messages"]
    
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
        
    llm = get_llm()
    tools = [search_company_knowledge]
    llm_with_tools = llm.bind_tools(tools)
    
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


# ─── Graph ───────────────────────────────────────────────────────────────────

def _route_by_intent(state: RAGState) -> str:
    return state["intent"]


def _should_use_tools(state: RAGState) -> str:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "end"


def _build_agent_graph():
    """Build and return the compiled Agentic RAG StateGraph with Intent Gating."""
    builder = StateGraph(RAGState)
    tools = [search_company_knowledge]
    
    # Nodes
    builder.add_node("classifier", _classify_intent)
    builder.add_node("greeting", _handle_greeting)
    builder.add_node("ambiguity", _handle_ambiguity)
    builder.add_node("agent", _call_knowledge_agent)
    builder.add_node("tools", ToolNode(tools=tools))
    
    # Flow: START -> classifier
    builder.add_edge(START, "classifier")
    
    # Router: classifier -> greeting OR ambiguity OR agent
    builder.add_conditional_edges(
        "classifier",
        _route_by_intent,
        {
            "GREETING": "greeting",
            "AMBIGUOUS": "ambiguity",
            "KNOWLEDGE": "agent"
        }
    )

    # Greeting and Ambiguity end immediately
    builder.add_edge("greeting", END)
    builder.add_edge("ambiguity", END)
    
    # Agent -> tools OR end
    builder.add_conditional_edges(
        "agent",
        _should_use_tools,
        {
            "tools": "tools",
            "end": END
        }
    )
    
    # After the tool executes, return back to the agent
    builder.add_edge("tools", "agent")
    
    return builder.compile()


@lru_cache(maxsize=1)
def get_rag_graph():
    """Return the singleton compiled Agentic RAG graph."""
    return _build_agent_graph()
