# Local RAG From Scratch

![rag_detail_v2](https://github.com/langchain-ai/rag-from-scratch/assets/122662504/54a2d76c-b07e-49e7-b4ce-fc45667360a1)

Retrieval-Augmented Generation (RAG) has emerged as a popular and powerful mechanism to expand an LLM's knowledge base. This technique uses documents retrieved from an external or local data source to ground the LLM's generation via in-context learning, providing relevant and up-to-date information.

In this repository, we demonstrate how to build a fully **local RAG system** from scratch using **Ollama**. This setup allows for **privacy-focused** and high-performance retrieval and generation without relying on external cloud services.

## Key Components of the Local RAG Setup:

- **Local Language Model**: Using **Ollama**, we load and run **llama models** **locally**. These models provide high-quality language generation without the need for **API** calls or **cloud-based services**.
  
- **Local Embeddings**: We use **Hugging Face**'s `sentence-transformers` to generate document **embeddings**. These embeddings are calculated **locally**, ensuring no data leaves your machine.
  
- **Local Retrieval (FAISS)**: For retrieval, we utilize Facebook AI Similarity Search (**FAISS**), a high-performance **vector** store that enables **fast** and **efficient** similarity searches. This ensures that relevant documents are retrieved quickly, supporting the **llama** model in producing grounded responses.

## Tutorial: Local RAG with LangChain on Google Colab

We have prepared a step-by-step tutorial to guide you through the process of building a Local RAG system using LangChain and Ollama on Google Colab. This tutorial will help you understand how to set up local language models, embeddings, and FAISS for retrieval-augmented generation.

### Click the button below to open the tutorial in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1GByk7ACuxQncfIcVpZAnOzhP7nU6JWwq)

