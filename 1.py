#!/usr/bin/env python
# coding: utf-8

# # Rag From Scratch: Overview
# 
# These notebooks walk through the process of building RAG app(s) from scratch, focusing on local models, embeddings, and efficient document retrieval.
# 
# ![RAG Overview](attachment:c566957c-a8ef-41a9-9b78-e089d35cf0b7.png)

# ## Environment Setup
# 
# ### (1) Installing Required Packages

# In[2]:
get_ipython().system('pip install langchain_community tiktoken langchain ollama faiss-cpu')


# ## Part 1: Running Local Models with Ollama and FAISS

# In[3]:
# Pull LLaMA 3.1 model using Ollama (local language model)
get_ipython().system('ollama pull llama3.1')


# In[4]:
# Import necessary libraries
import ollama
from langchain.llms import Ollama
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# Load the local LLaMA 3.1 model using Ollama
llm = Ollama(model="llama3.1")

# Use local embeddings for document retrieval
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Sample documents
texts = ["Sample text 1", "Sample text 2", "Sample text 3"]

# Create the FAISS vectorstore locally
faiss_index = FAISS.from_texts(texts, embeddings)

# Function to create the RAG pipeline
def local_rag(question, faiss_index, llm):
    # Retrieve relevant documents
    docs = faiss_index.similarity_search(question)
    
    # Create the prompt with retrieved documents
    prompt = f"Context: {docs}\n\nQuestion: {question}\n\nAnswer:"
    
    # Get the model response
    response = llm(prompt)
    
    return response


# In[5]:
# Run the RAG pipeline
question = "What is Retrieval-Augmented Generation (RAG)?"
response = local_rag(question, faiss_index, llm)

# Output the response
print(response)


# ## Part 2: Efficient Document Indexing with FAISS

# In[6]:
# Load the Hugging Face Embeddings for local use
from langchain.embeddings import HuggingFaceEmbeddings

# Example documents
question = "What kinds of pets do I like?"
document = "My favorite pet is a cat."

# Embed the query and document using Hugging Face embeddings
embd = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
query_result = embd.embed_query(question)
document_result = embd.embed_query(document)

# Calculate the length of the query embeddings
len(query_result)


# In[7]:
# Implement cosine similarity for comparing query and document embeddings
import numpy as np

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

similarity = cosine_similarity(query_result, document_result)
print("Cosine Similarity:", similarity)


# ## Part 3: Document Splitting and Vectorization

# In[8]:
# Load blog posts and use RecursiveCharacterTextSplitter for splitting documents
from langchain_community.document_loaders import WebBaseLoader
import bs4

# Load blog from web
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(class_=("post-content", "post-title", "post-header"))
    )
)
blog_docs = loader.load()

# Split the blog documents
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=300, 
    chunk_overlap=50
)
splits = text_splitter.split_documents(blog_docs)


# In[9]:
# Use FAISS for local vector storage and retrieval
texts = [doc.page_content for doc in splits]
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Create FAISS vectorstore
vectorstore = FAISS.from_texts(texts=texts, embedding=embedding_model)

# Convert the vectorstore into a retriever with search options
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})


# In[10]:
# Test document retrieval
docs = retriever.get_relevant_documents("What is Task Decomposition?")
print("Number of relevant documents:", len(docs))


# ## Part 4: Generating Responses with Local LLaMA

# In[11]:
