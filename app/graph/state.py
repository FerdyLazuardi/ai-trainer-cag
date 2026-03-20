"""
RAG pipeline state definition for LangGraph.
All nodes read from and write to this shared typed state.
"""
import operator
from typing import Annotated, Any, List, Optional, TypedDict

from langchain_core.messages import AnyMessage


class RAGState(TypedDict):
    """
    Shared state for the Optimized RAG pipeline.
    """
    messages: Annotated[list[AnyMessage], operator.add]
    query: str
    conversation_id: str
    intent: Optional[str]
    error: Optional[str]
    # Holds retrieved chunks from rag_node, consumed by generate_node
    retrieved_context: Optional[List[dict]]
