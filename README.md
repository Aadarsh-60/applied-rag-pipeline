# RAG from Scratch 

A beginner-friendly implementation of Retrieval-Augmented Generation (RAG) built step-by-step using LangChain, FAISS, HuggingFace embeddings, and **Groq (Llama 3)**. 

This project has been upgraded with a **Streamlit Web UI**, allowing you to dynamically upload documents directly from your browser, watch the backend chunk and embed them in real-time, and chat with them instantly!

---

## What is RAG and Why Does It Matter?

**The problem with plain LLMs:**
Large Language Models can generate impressive answers, but they have limitations. They may hallucinate, rely on outdated knowledge, or fail to answer questions about private documents such as research papers, internal company policies, product manuals, or project-specific notes.

**What RAG does:**
Retrieval-Augmented Generation solves this by giving the LLM access to external documents at query time. Instead of retraining the model, the system retrieves the most relevant chunks from your documents and sends them to the LLM as context. The LLM then generates an answer grounded in those retrieved chunks.

**Why it matters:**
RAG is one of the most practical architectures for building production-ready AI Q&A systems. It is cost-effective, easy to update, and more transparent because the system can show which document chunks were used to generate the answer.

This project demonstrates the core RAG workflow from scratch:

```text
Load documents → Chunk text → Generate embeddings → Store in FAISS → Retrieve relevant chunks → Generate answer with Groq (Llama 3)
```

---

## Architecture

```text
UPLOAD VIA BROWSER (PDF / TXT / DOCX)
         │
         ▼
  ┌─────────────┐
  │  1. LOAD    │  Read files from disk into LangChain Document objects
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │  2. CHUNK   │  Split large documents into smaller overlapping chunks
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │  3. EMBED   │  Convert each chunk into vector embeddings
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │  4. INDEX   │  Store vectors in FAISS for similarity search
  └──────┬──────┘
         │
         │                    USER QUESTION IN WEB UI
         │                         │
         │                         ▼
         │                  ┌─────────────┐
         │                  │  5. EMBED   │  Convert question into a vector
         │                  └──────┬──────┘
         │                         │
         ▼                         ▼
  ┌─────────────────────────────────────┐
  │         FAISS SIMILARITY SEARCH     │  Retrieve top-k relevant chunks
  └──────────────────┬──────────────────┘
                     │
                     ▼
             RELEVANT DOCUMENT CHUNKS
                     │
                     ▼
  ┌─────────────────────────────────────┐
  │  6. GENERATE (Groq Llama 3)         │  Generate grounded answer
  └──────────────────┬──────────────────┘
                     │
                     ▼
              SOURCE-GROUNDED ANSWER ✅
```

---

## Tech Stack

| Component        | Library / Tool                     | Purpose                                       |
| ---------------- | ---------------------------------- | --------------------------------------------- |
| Frontend UI      | `streamlit`                        | Dynamic web interface for uploading and chat  |
| Document loading | `langchain-community` loaders      | Read PDF, TXT, and DOCX files                 |
| Text splitting   | `langchain-text-splitters`         | Split documents into overlapping chunks       |
| Embeddings       | `sentence-transformers`            | Convert text into vector embeddings           |
| Vector database  | `faiss-cpu`                        | Store and search document embeddings          |
| LLM              | Groq (Llama 3 API)                 | Generate final answers from retrieved context |
| Env management   | `python-dotenv`                    | Load API keys from `.env`                     |
| Orchestration    | LangChain                          | Connect retrieval and generation steps        |

---

## Step-by-Step Setup

### 1. Clone the repository

```bash
git clone https://github.com/pranjalisr/Applied-RAG-Systems.git
cd Applied-RAG-Systems/01-rag-from-scratch
```

### 2. Create and activate a virtual environment

Use **Python 3.11**.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Verify the Python version:

```bash
python --version
```

### 3. Install dependencies

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install all dependencies:

```bash
python -m pip install -r requirements.txt
python -m pip install streamlit langchain_text_splitters
```

> First install may take a few minutes because `faiss-cpu` and `sentence-transformers` are large packages.

---

## Configure Groq API

This project uses the lightning-fast **Groq API (Llama 3)** for answer generation.

Create a `.env` file:

```bash
cp .env.example .env
```

Open `.env` and add your keys (you can get a free key from console.groq.com):

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_BASE_URL=https://api.groq.com/openai/v1
GROQ_MODEL=llama-3.1-8b-instant
```

Do not wrap the values in quotes.

---

## Run the Project (Web UI)

The best way to experience this project is through the newly added **Streamlit Web UI**.

```bash
python -m streamlit run app.py
```

1. It will open a browser tab at `http://localhost:8501`.
2. Open the left sidebar and use the **Document Upload** widget to upload any `.pdf`, `.txt`, or `.docx` files.
3. Click **Process Documents**.
4. You will see the live terminal logs inside the UI as it chunks and embeds your documents.
5. Once loaded, you can ask questions about your documents in the main chat!

*Note: The UI allows you to add multiple files dynamically without needing to touch the backend folders!*

---

## Run the Project (CLI Mode)

If you prefer the terminal, the original CLI still works perfectly!

Drop any `.pdf`, `.txt`, or `.docx` files into:
```text
data/sample_docs/
```

### Interactive mode
```bash
python main.py --model groq-llama3
```

### Single question mode
```bash
python main.py --model groq-llama3 --question "What problems do LLMs face?"
```

### Debug mode
```bash
python main.py --debug --model groq-llama3 --question "Explain how RAG works."
```
Debug mode shows the retrieved chunks and the final prompt sent to the LLM. 

*(If you change documents in the `data/sample_docs/` folder, remember to `rm -rf faiss_index` before running `main.py` again).*

---

## Supported File Types

| File type | Support | Notes                                       |
| --------- | ------- | ------------------------------------------- |
| `.pdf`    | ✅       | Each page can be loaded as document content |
| `.txt`    | ✅       | Simple text files are supported             |
| `.docx`   | ✅       | Word documents are supported                |
| `.csv`    | ❌       | Not supported in this version               |

---

## Prompt Template

A strict prompt is used to reduce hallucinations:

```python
RAG_PROMPT_TEMPLATE = """
You are a helpful assistant. Answer the question based ONLY on the following context.
If the answer is not in the context, say "I don't know based on the provided documents."
Do not use your general knowledge.
Do not add a "What is not covered" section unless the user specifically asks for limitations.
Give a clean, structured answer with headings and bullet points.

Context:
{context}

Question: {question}

Answer:
"""
```

This keeps the answer grounded in the retrieved document chunks.

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'langchain_text_splitters'` or `'streamlit'`
Make sure you install the new frontend dependencies!
```bash
python -m pip install streamlit langchain_text_splitters
```

### Streamlit `CacheReplayClosureError`
If you see an error about caching Streamlit elements, ensure you are running `app.py` from within your activated virtual environment using:
```bash
python -m streamlit run app.py
```

### `No matching distribution found for faiss-cpu==1.8.0`
This usually happens when using Python 3.14 or later. Recreate your virtual environment using **Python 3.11**.

### Groq API Error (400 or 402)
If you get `Insufficient Balance` or `Model Decommissioned`, verify that your `GROQ_MODEL` is set to `llama-3.1-8b-instant` in your `.env` file, and that your API key is correct.

---

## Final Notes

This project shows the complete foundation of a RAG-based document question-answering system.

The main idea is simple:

```text
Do not rely only on the LLM's internal memory.
Retrieve relevant external knowledge first.
Then generate an answer grounded in that retrieved context.
```

This version uses:
* **HuggingFace embeddings** for local vector creation (free and private)
* **FAISS** for rapid vector search
* **Groq (Llama 3)** for lightning-fast answer generation
* **Streamlit** for a beautiful, dynamic user interface
* **LangChain** for orchestration

It is a practical starting point for building document Q&A systems, research assistants, internal knowledge bots, and production GenAI applications.
