# Local RAG From Scratch

LLMs are typically trained on a large but fixed corpus of data, limiting their ability to reason about private or recent information. Fine-tuning can mitigate this to some extent but is often [not well-suited for factual recall](https://www.anyscale.com/blog/fine-tuning-is-for-form-not-facts) and [can be costly](https://www.glean.com/blog/how-to-build-an-ai-assistant-for-the-enterprise).

Retrieval-Augmented Generation (RAG) has emerged as a popular and powerful mechanism to expand an LLM's knowledge base. This technique uses documents retrieved from an external or local data source to ground the LLM's generation via in-context learning, providing relevant and up-to-date information.

In this repository, we demonstrate how to build a fully **local RAG system** from scratch using **Ollama**. This setup allows for **privacy-focused** and high-performance retrieval and generation without relying on external cloud services.

## Key Components of the Local RAG Setup:

- **Local Language Model**: Using **Ollama**, we load and run **llama models** **locally**. These models provide high-quality language generation without the need for **API** calls or **cloud-based **services.
  
- **Local Embeddings**: We use **Hugging Face**'s `sentence-transformers` to generate document **embeddings**. These embeddings are calculated **locally**, ensuring no data leaves your machine.
  
- **Local Retrieval (FAISS)**: For retrieval, we utilize Facebook AI Similarity Search (**FAISS**), a high-performance **vector** store that enables **fast** and **efficient** similarity searches. This ensures that relevant documents are retrieved quickly, supporting the **llama** model in producing grounded responses.

## Running the Project Locally

1. **Install Ollama**: Follow the installation instructions on the [Ollama website](https://ollama.com/) to set up the tool locally.

2. **Pull a LLaMA 3.1 Model**: Once Ollama is installed, you can download any version of the LLaMA 3.1 models locally.
    ```bash
    ollama pull llama3.1
    ```
