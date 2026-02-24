🛍️ Domain-Restricted E-Commerce RAG Assistant
📌 Project Overview

This project implements a Retrieval-Augmented Generation (RAG) based conversational assistant for an E-Commerce support domain.

The assistant answers questions strictly using uploaded business documents (return policies, warranties, product manuals) and refuses to respond when information is not available in the knowledge base.

The system demonstrates:

Context-restricted generation

Similarity-based retrieval gating

Source citation

Conversational memory

Confidence scoring

Clean modular architecture

Streamlit-based UI

🏗️ System Architecture
High-Level Flow

Documents → Chunking

Chunked text → Embeddings

Embeddings → FAISS Vector Store

User Query → Similarity Search

Retrieved Context → LLM (gpt-4o-mini)

Response + Sources + Confidence

Architecture Components
1️⃣ Ingestion Layer (ingest.py)

Loads domain documents

Splits into chunks using text splitter

Converts to embeddings (text-embedding-3-small)

Stores vectors in FAISS

2️⃣ Retrieval + LLM Core (rag_core.py)

Responsible for:

Similarity search

Threshold filtering (hallucination prevention)

Prompt control

Conversational memory

Response generation

Confidence score calculation

3️⃣ User Interface (app.py)

Built using Streamlit:

Chat-style interface

Displays sources

Shows confidence score

Maintains session memory

🛡️ Hallucination Prevention Strategy

The system prevents hallucinations using:

Strict prompt instructions

Similarity threshold filtering

Domain-restricted answering

Explicit refusal message when context not found

Temperature set to 0 for deterministic outputs

If similarity score exceeds threshold, system responds:

"I don’t have enough information in the provided documents."

📊 Confidence Score Logic

Confidence = 1 - similarity_score

Lower similarity distance → Higher confidence.

This improves transparency and explainability.

Tech Stack

LangChain

OpenAI (GPT-4o-mini)

FAISS

Streamlit

Python

Here are publicly available documents (or official policy pages you can save as PDF) that you can use for your RAG project in the E-Commerce – Product manuals & return policies domain:

📄 1. Best Buy Return & Exchange Policy (PDF)

This is an official return & exchange policy from Best Buy — useful as a document for your vector store.

🔗 Best Buy Return & Exchange Policy PDF
https://partners.bestbuy.com/documents/20126/3029894/Return%2B%26%2BExchange%2BPolicy.pdf/ee165181-38ed-21af-be39-30c2c7b34597?t=1629819812060

📄 2. Samsung Returns and Faulty Goods Policy (PDF)

Official Samsung policy for returns and faulty goods — you can download this PDF for ingestion.

🔗 Samsung Returns and Faulty Goods Policy PDF
https://images.samsung.com/is/content/samsung/assets/uk/returns-policy/Returns_and_faulty_goods_statement_FINAL_160522.pdf

📄 3. Apple Returns & Refunds Policy (Web Page)

Apple’s official returns/refunds support page — good for saving as PDF.

🔗 Apple Returns & Refunds Policy (Apple Support)
https://www.apple.com/shop/help/returns_refund