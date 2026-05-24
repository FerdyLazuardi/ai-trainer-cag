# AI Learning Assistant for Amartha LMS

### Video Highlight

## **Project Overview**

The AI Learning Assistant (peped-BE) is the AI-powered backend backbone developed for Amartha’s Learning Management System (Amarthapedia). The core challenge was managing vast, scattered internal knowledge bases and transforming them into a responsive, intelligent assistant. This project focuses on building an efficient Retrieval-Augmented Generation (RAG) system to ensure every employee (the A-Team) can access accurate, highly relevant information instantly through natural conversation.

### **1. Strategic Objectives**

The primary goal of developing this intelligent assistant was to democratize knowledge access within the company. Key strategic objectives included:

- **Enhance Knowledge Accuracy:** Ensure the assistant strictly grounds its answers in valid internal data to effectively minimize AI hallucinations.
- **Optimize Response Speed:** Reduce technical friction and retrieval latency so the learning flow is never interrupted by long wait times.
- **Empower On-Demand Learning:** Provide the A-Team with instant, 24/7 learning support, helping them overcome study blockers and understand complex materials independently without waiting for human intervention

### **2. Engineering Strategy & Architecture**

To build a highly reliable and intelligent assistant, I utilized an Agentic RAG approach backed by a modern, high-performance tech stack:

- **Agentic Orchestration (LangGraph & FastAPI):** Leveraged FastAPI for high-performance APIs and LangGraph to manage complex AI workflows, enabling the assistant to "think" and reason before generating an answer.
- **Advanced Retrieval (Qdrant & LightRAG):** Implemented Qdrant as the primary vector database for precise semantic search, alongside research into LightRAG to improve the accuracy of information retrieval from dense knowledge bases.
- **Performance Engineering (Redis Caching):** Utilized Redis for semantic caching, which significantly reduces the LLM workload and accelerates response times for frequently asked or similar questions.
- **Self-Hosted Infrastructure:** Prioritized data security and cost-efficiency by deploying services like Langfuse and rerankers on a local Proxmox server using Docker.

## **Summary of Core Features**

- **Semantic Search & Retrieval:** The assistant's ability to understand the deeper context and intent of a user's query, moving far beyond simple keyword matching.
- **Long-Term Agent Memory:** Integration of agent-managed long-term memory (via Qdrant) to provide a continuous, highly personalized learning experience.
- **Seamless LMS Integration:** A robust backend architecture designed to integrate smoothly with the Amarthapedia frontend, ensuring flawless data transition between systems.
