"""LLM-as-judge evaluation for A-Pedi RAG output quality.

Eval is async (arq), sampled, and writes scores to Phoenix as span
annotations — same channel as intent_scores and retriever scores. This
keeps observability single-stack (Phoenix UI / REST) without a parallel
metric store.
"""
