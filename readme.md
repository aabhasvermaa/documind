# DocuMind: RAG-based Document Analysis AI

## Project Overview
This project implements a Retrieval-Augmented Generation (RAG) pipeline to chat with PDF documents. It allows users to ask questions about a specific document and receive accurate answers based only on that document's content, reducing AI hallucinations.

## Tech Stack
* LLM: Google Gemini 1.5 Flash (via LangChain)
* Vector DB: ChromaDB
* Embeddings: HuggingFace (all-MiniLM-L6-v2)
* Orchestration: LangChain

## Key Learnings
1. Vector Embeddings: How to convert text into numbers for semantic search.
2. Context Window Management: Chunking large documents to fit into LLM memory.
3. Prompt Engineering: Constraining the AI to answer strictly from provided data.