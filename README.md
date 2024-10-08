# Local RAG with LangChain Tutorial

![rag_detail_v2](https://github.com/langchain-ai/rag-from-scratch/assets/122662504/54a2d76c-b07e-49e7-b4ce-fc45667360a1)

Retrieval-Augmented Generation (RAG) has emerged as a robust technique for enhancing the knowledge base of Language Learning Models (LLMs). By retrieving documents from local or external sources, RAG enables models to provide relevant, up-to-date, and grounded responses through in-context learning.

This repository showcases how to build a fully **local RAG system** from scratch using **Ollama** and **LangChain**. Our focus is on **privacy** and **performance**, as all components—language models, embeddings, and retrieval—are processed locally, eliminating reliance on external cloud services.

## Key Components of the Local RAG Setup

- **Local Language Model**: Utilizing **Ollama**, we run **llama models** locally on your machine. This setup avoids API calls and cloud services, keeping your data private while providing high-quality language generation.
  
- **Local Embeddings**: We generate document **embeddings** using **Hugging Face's** `sentence-transformers`, all performed locally, ensuring that your data never leaves your environment.

- **Local Retrieval (FAISS)**: To retrieve relevant documents, we employ **Facebook AI Similarity Search (FAISS)**, a high-performance vector store that facilitates efficient similarity searches. This ensures the retrieved documents are used to ground the llama model's responses.

## Local RAG with LangChain on Google Colab

We offer a series of Google Colab tutorials that walk you through building a Local RAG system with **LangChain** and **Ollama**. These tutorials cater to both CPU and GPU setups, allowing you to leverage the resources available on your machine.

### Google Colab Tutorials:

I have shared the folder containing my Google Colab notebooks:

[Google Colab Notebooks Folder](https://drive.google.com/drive/folders/16jfiLFQL8_yjZf29Au8LPCyc7yXe0C4w?usp=sharing)

