"""LLM-as-judge evaluation for Ava RAG output quality.

Eval is async (arq), sampled, and writes faithfulness scores to Postgres
(agent_logs) so the Streamlit dashboard can surface low-faithfulness turns.
"""
