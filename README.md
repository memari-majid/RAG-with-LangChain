# Local RAG From Scratch

![rag_detail_v2](https://github.com/langchain-ai/rag-from-scratch/assets/122662504/54a2d76c-b07e-49e7-b4ce-fc45667360a1)

Retrieval-Augmented Generation (RAG) has become a powerful technique for expanding the knowledge base of a Language Learning Model (LLM). This approach uses documents retrieved from local or external sources to provide grounded, relevant, and up-to-date information, which enhances the LLM's generation through in-context learning.

This repository demonstrates how to build a fully **local RAG system** from scratch using **Ollama**. The emphasis here is on ensuring **privacy** and **performance**, as the entire process is run locally, without the need for external cloud services.

## Key Components of the Local RAG Setup

- **Local Language Model**: We use **Ollama** to run **llama models** entirely on your local machine. This eliminates the need for **API** calls or **cloud-based services**, ensuring privacy while delivering high-quality language generation.
  
- **Local Embeddings**: Using **Hugging Face**'s `sentence-transformers`, we generate document **embeddings** locally. This ensures that no data leaves your machine and all processing stays within your control.

- **Local Retrieval (FAISS)**: For fast and efficient document retrieval, we utilize **Facebook AI Similarity Search (FAISS)**. FAISS is a high-performance vector store that performs similarity searches, ensuring that relevant documents are retrieved quickly to support grounded responses by the **llama** model.

## Tutorial: Local RAG with LangChain on Google Colab

We have prepared several Google Colab tutorials to guide you through the process of building a Local RAG system using **LangChain** and **Ollama**. These tutorials cover both CPU and GPU setups, allowing you to adapt to the resources available on your local machine.

### Click the button below to open the tutorial in Google Colab:

### Local RAG with CPU

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1GByk7ACuxQncfIcVpZAnOzhP7nU6JWwq)

This tutorial provides a step-by-step guide for setting up a **local RAG pipeline** using CPU-based computation. It's ideal for users who don’t have access to GPU resources.

### Local RAG with GPU

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1MO4YLQ3kkA5_eSyzgRz4wlPq2LFeobaV)

This tutorial demonstrates how to accelerate your **local RAG pipeline** using **GPU** resources. It includes the use of GPU-based Hugging Face embeddings and FAISS for faster vector similarity searches.

### Additional Google Colab Tutorials:

- **Simple Local RAG with CPU**: [Open in Colab](https://colab.research.google.com/drive/1-Hq9l2E7tWwOt6WfkaISMMnI1o7MIaRg)
  
- **In-depth Local RAG with CPU**: [Open in Colab](https://colab.research.google.com/drive/1rPIPrH0m9b4tQzzqJ9ZjuZfaNpnfWnnR)

- **Simple Local RAG with GPU and Large Memory**: [Open in Colab](https://colab.research.google.com/drive/1MO4YLQ3kkA5_eSyzgRz4wlPq2LFeobaV)

- **LangChain Agents**: [Open in Colab](https://colab.research.google.com/drive/1-mFnpOJjxErMm-yht625CmjayqsEFjo4)

- **Advanced RAG with Capability to Read Different File Types and Databases**: [Open in Colab](https://colab.research.google.com/drive/1fVNBMgjf6yQX0qpGTd2RF6HR-agjoppN)


