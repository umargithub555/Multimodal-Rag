# 📚 Multimodal RAG System

A production-grade Multimodal Retrieval-Augmented Generation (RAG) system that can analyze text + images from documents (PDFs, books, lecture notes), generate embeddings, store them in a vector DB, and answer user queries using a multimodal LLM.

This project is designed with real-world developer practices, focusing on scalability, modularity, and maintainability.

## 🚀 Features

🔍 1. Text Chunking & Summarization

Extract text from large documents.

Smart chunking using semantic boundaries.

Generate concise summaries for each chunk.

🖼️ 2. Image Extraction & Vision Summaries

Detect and extract images from PDFs.

Use multimodal LLM to generate meaning-rich image summaries.

Convert these summaries into embeddings.

🧠 3. Embeddings + Vector Database (Chroma/FAISS)

Use Nomic, OpenAI, or Google embeddings.

Store text + image vectors in a vector database.

Metadata includes:

Chunk text

Image summary

Image binary path

Page number

💬 4. Multimodal User Query Handling

User enters text (and optionally uploads an image).

Query is converted to embeddings.

Find top relevant text chunks + images via vector search.

Multimodal LLM generates final answer using both.

⚙️ 5. Modular Production Architecture

Pipeline-based structure

Clear separation of

Ingestion

Preprocessing

Embedding

Retrieval

Response generation



## Installation
```
git clone <repo-url>

pip install -r requirements.txt

** Additional Requirements

Install or download Poppler and Pytesseract add to path

POPPLER_PATH = r"C:\poppler-24.08.0\Library\bin"

TESSERACT_PATH=r"C:\Program Files\Tesseract-OCR"

os.environ["PATH"] += os.pathsep + POPPLER_PATH  

os.environ["PATH"] += os.pathsep + TESSERACT_PATH

```

### Create a .env file:
```
GOOGLE_API_KEY=your_key
OPENAI_API_KEY=your_key
VECTOR_DB_PATH=./output/vector_store
ELEVEN_LAB_KEY=your_key
```



# Project Overview
This repository serves as a rough, functional implementation of a complete Multimodal RAG (Retrieval-Augmented Generation) system, enhanced with:

Quiz generation powered by regex-based or semantic chapter extraction

Mock interview/Test workflows with integrated Text-to-Speech (TTS)

A flexible multimodal pipeline capable of handling text + images

Note: This codebase is intentionally kept simple for learning and experimentation. The version deployed in production for a client uses a more refined architecture, improved abstractions, and stricter modularization.


## Important Note for Users

To fully understand and extend this project, users are strongly encouraged to study each file individually.
Every module has been designed with clear responsibilities, and understanding these components is essential before integrating or modifying any advanced features.

** Use the LangChain version: "0.3.27" for using MultiVectorRetrieval
