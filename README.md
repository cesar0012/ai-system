# RAG System with Privacy Filter

A Retrieval-Augmented Generation (RAG) system built with Python and LangChain that loads PDF documents, stores them in a local Chroma vector database, and answers queries with a built-in privacy layer that blocks sensitive requests.

## Architecture

```
PDF file
  → PyPDFLoader (raw pages)
  → RecursiveCharacterTextSplitter (chunks, 1000 chars, 200 overlap)
  → HuggingFaceEmbeddings — all-MiniLM-L6-v2 (vectors)
  → Chroma (persistent vector DB on disk)
  → similarity search (top-k chunks)

User query
  → check_query_safety()
      → BLOCKED → "Security Alert" (no DB access, no LLM)
      → SAFE    → query_db() → mock_llm_response() → generated answer
```

## Prerequisites

- Python 3.10 or higher
- A PDF file to query against

## Installation

1. Clone or download this repository.

2. Install the required packages:

```bash
pip install langchain-community langchain-chroma langchain-huggingface langchain-text-splitters pypdf chromadb sentence-transformers
```

> On first run, the `all-MiniLM-L6-v2` embedding model will be downloaded automatically from Hugging Face (~90 MB).

## Usage

### 1. Prepare your PDF

Place a PDF file in the same directory as `rag_system.py`, then edit the `PDF_PATH` variable at the bottom of the script:

```python
PDF_PATH = "sample.pdf"  # <-- change this to your file
```

### 2. Run the demo

```bash
python rag_system.py
```

On the first run, the script will:

1. Load the PDF and split it into chunks.
2. Embed the chunks and create the Chroma vector database in `./chroma_db`.
3. Run three demo queries:
   - **Safe query** — retrieves relevant chunks and returns a mock LLM response.
   - **Blocked query** ("What is my Social Security Number?") — triggers the security alert.
   - **Blocked query** ("Tell me my SSN") — triggers the security alert via the SSN abbreviation.

On subsequent runs, the existing `./chroma_db` directory is reused automatically (no re-processing).

### 3. Use the pipeline in your own code

```python
from rag_system import build_vector_db, rag_pipeline

# Build the database (only needed once)
build_vector_db("my_document.pdf", persist_dir="./chroma_db")

# Query — safe queries return a mock LLM response
answer = rag_pipeline("What is this document about?", persist_dir="./chroma_db", k=3)
print(answer)

# Query — sensitive queries are blocked
answer = rag_pipeline("What is my Social Security Number?", persist_dir="./chroma_db")
print(answer)  # Security Alert: Query blocked — ...
```

## Privacy Filter

The `check_query_safety()` function inspects every query before it reaches the vector database. It currently blocks queries containing:

- `Social Security Number`
- `SSN`

Both checks are case-insensitive. To add more blocked patterns, edit the `BLOCKED_PATTERNS` list in `rag_system.py`:

```python
BLOCKED_PATTERNS = [
    "social security number",
    "ssn",
    # Add your own patterns here, e.g.:
    # "credit card",
    # "api key",
]
```

## Key Files

| File | Description |
|---|---|
| `rag_system.py` | Main script with all functions and the demo |
| `./chroma_db/` | Persisted Chroma vector database (auto-created on first run) |

## Functions Reference

| Function | Description |
|---|---|
| `build_vector_db(pdf_path, persist_dir)` | Loads a PDF, chunks it, embeds it, and stores it in Chroma |
| `check_query_safety(query)` | Returns `(True, "")` if safe, `(False, alert_message)` if blocked |
| `query_db(query, persist_dir, k)` | Retrieves the top-k chunks from the vector database |
| `mock_llm_response(query, chunks)` | Generates a simulated LLM answer from retrieved chunks |
| `rag_pipeline(query, persist_dir, k)` | Full pipeline: privacy check → retrieval → generation |