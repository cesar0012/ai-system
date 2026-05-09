"""
RAG System using LangChain + Chroma + HuggingFace Embeddings

Data flow overview:
    PDF file → PyPDFLoader (raw pages) → RecursiveCharacterTextSplitter (chunks)
    → HuggingFaceEmbeddings (vectors) → Chroma (persistent vector DB)
    → similarity search (top-k chunks) → mock LLM response

    User query → check_query_safety()
        → BLOCKED → return "Security Alert" (stop, no DB query, no LLM)
        → SAFE    → query_db() → mock_llm_response() → return generated answer

Dependencies:
    pip install langchain-community langchain-chroma langchain-huggingface langchain-text-splitters pypdf chromadb sentence-transformers
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# ---------------------------------------------------------------------------
# 1. Build the vector database from a PDF
# ---------------------------------------------------------------------------
def build_vector_db(pdf_path: str, persist_dir: str = "./chroma_db") -> Chroma:
    """
    Load a PDF, split it into overlapping chunks, embed each chunk, and
    persist the result in a local Chroma vector database.

    Args:
        pdf_path:   Path to the source PDF file.
        persist_dir: Directory where Chroma will store its data on disk.

    Returns:
        A Chroma vector-store instance that is ready for querying.
    """

    # Step 1 – Load every page of the PDF as separate Document objects.
    # Each Document carries the page text plus metadata (page number, source).
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    # Step 2 – Split the pages into smaller chunks so that each chunk is
    # small enough to embed efficiently and relevant enough for accurate
    # retrieval.  The 200-character overlap ensures context is not lost
    # at chunk boundaries.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(pages)

    # Step 3 – Create the embedding model that will convert text chunks
    # into dense vectors.  all-MiniLM-L6-v2 produces 384-dim embeddings
    # and is a good trade-off between speed and quality.
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
    )

    # Step 4 – Build (and persist) the Chroma vector store.
    # Chroma embeds every chunk, stores the vectors, and saves the index
    # to persist_dir so it can be reloaded later without re-processing.
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
    )

    print(f"Vector DB created: {len(chunks)} chunks stored in '{persist_dir}'")
    return vector_db


# ---------------------------------------------------------------------------
# 2. Privacy / security filter
# ---------------------------------------------------------------------------
# Patterns that indicate a sensitive query.  Extend this list with regex
# patterns or plain strings as needed (e.g. credit-card numbers, API keys).
BLOCKED_PATTERNS = [
    "social security number",
    "ssn",
]


def check_query_safety(query: str) -> tuple[bool, str]:
    """
    Inspect the user's query for sensitive content.

    Args:
        query: The raw user query string.

    Returns:
        A tuple (is_safe, message):
            is_safe  = True  → the query is clean, proceed with retrieval.
            is_safe  = False → the query is blocked; message explains why.
            message  = "" when safe, or a "Security Alert" string when blocked.
    """
    query_lower = query.lower()

    for pattern in BLOCKED_PATTERNS:
        if pattern in query_lower:
            return (
                False,
                f"Security Alert: Query blocked — detected sensitive pattern "
                f"'{pattern}'. Request denied.",
            )

    return True, ""


# ---------------------------------------------------------------------------
# 3. Query the vector database
# ---------------------------------------------------------------------------
def query_db(
    query: str,
    persist_dir: str = "./chroma_db",
    k: int = 3,
) -> list:
    """
    Load an existing Chroma database and return the top-k most relevant
    chunks for the given query.

    Args:
        query:       The user's natural-language question.
        persist_dir: Directory where the Chroma database is stored.
        k:           Number of top results to return.

    Returns:
        A list of LangChain Document objects sorted by relevance (best first).
        Each Document has:
            .page_content  – the chunk text
            .metadata      – dict with source file, page number, etc.
    """

    # Re-create the embedding model (must match the one used during build)
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
    )

    # Load the persisted vector store from disk
    vector_db = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
    )

    # Perform similarity search: the query is embedded, compared against
    # all stored vectors, and the k closest chunks are returned.
    results = vector_db.similarity_search(query, k=k)

    return results


# ---------------------------------------------------------------------------
# 4. Mock LLM response generator
# ---------------------------------------------------------------------------
def mock_llm_response(query: str, chunks: list) -> str:
    """
    Simulate a generative LLM response by combining the retrieved chunks
    with the user's query into a formatted answer string.

    In a production system this function would call an actual LLM (e.g.
    OpenAI, Claude, local model) with a prompt that includes the chunks
    as context.  Here we simply format the context and query together.

    Args:
        query:  The user's original question.
        chunks: The list of Document objects returned by query_db().

    Returns:
        A string representing the "generated" answer.
    """
    # Combine the text of all retrieved chunks into a single context block
    context_parts = []
    for i, doc in enumerate(chunks, 1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        context_parts.append(f"[Context {i} — source: {source}, page: {page}]\n{doc.page_content}")

    context_block = "\n\n".join(context_parts)

    # Build the mock response — in production this would be the LLM output
    response = (
        f"Based on the provided context, here is the answer to: \"{query}\"\n\n"
        f"--- Retrieved Context ---\n{context_block}\n--- End of Context ---"
    )

    return response


# ---------------------------------------------------------------------------
# 5. Full RAG pipeline (privacy filter → retrieval → generation)
# ---------------------------------------------------------------------------
def rag_pipeline(
    query: str,
    persist_dir: str = "./chroma_db",
    k: int = 3,
) -> str:
    """
    End-to-end RAG pipeline with privacy guard.

    1. Check the query for sensitive content.
       → If blocked, return a Security Alert immediately (no DB access).
    2. Retrieve the top-k relevant chunks from the vector DB.
    3. Pass the chunks to the (mock) LLM and return the generated answer.

    Args:
        query:       The user's natural-language question.
        persist_dir: Directory where the Chroma database is stored.
        k:           Number of chunks to retrieve.

    Returns:
        Either a "Security Alert" string (if blocked) or the generated
        answer string from the mock LLM.
    """
    # Step 1 – Privacy gate
    is_safe, alert_message = check_query_safety(query)
    if not is_safe:
        return alert_message

    # Step 2 – Retrieve relevant chunks
    chunks = query_db(query, persist_dir, k=k)

    # Step 3 – Generate response from chunks
    response = mock_llm_response(query, chunks)

    return response


# ---------------------------------------------------------------------------
# 6. Main block – end-to-end demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import os

    # --- Configuration -------------------------------------------------------
    PDF_PATH = "sample.pdf"          # <-- change this to your PDF file
    PERSIST_DIR = "./chroma_db"
    # -------------------------------------------------------------------------

    # Build the vector DB (skip if it already exists on disk)
    if not os.path.isdir(PERSIST_DIR):
        if not os.path.isfile(PDF_PATH):
            print(f"Error: PDF not found at '{PDF_PATH}'.")
            print("Place a PDF file next to this script or edit PDF_PATH.")
            sys.exit(1)
        print("Building vector database …")
        build_vector_db(PDF_PATH, PERSIST_DIR)
    else:
        print(f"Reusing existing vector DB at '{PERSIST_DIR}'.")

    # ---- Demo: safe query (should produce a normal answer) ------------------
    SAFE_QUERY = "What is this document about?"

    print("\n" + "=" * 70)
    print("DEMO 1 — Safe query (should pass the privacy filter)")
    print("=" * 70)
    print(f"Query: {SAFE_QUERY}\n")

    result = rag_pipeline(SAFE_QUERY, PERSIST_DIR, k=3)
    print(result)

    # ---- Demo: blocked query (should trigger the security alert) ------------
    BLOCKED_QUERY = "What is my Social Security Number?"

    print("\n" + "=" * 70)
    print("DEMO 2 — Blocked query (should trigger Security Alert)")
    print("=" * 70)
    print(f"Query: {BLOCKED_QUERY}\n")

    result = rag_pipeline(BLOCKED_QUERY, PERSIST_DIR, k=3)
    print(result)

    # ---- Also test the SSN abbreviation variant -----------------------------
    SSN_QUERY = "Tell me my SSN"

    print("\n" + "=" * 70)
    print("DEMO 3 — Blocked query (SSN abbreviation)")
    print("=" * 70)
    print(f"Query: {SSN_QUERY}\n")

    result = rag_pipeline(SSN_QUERY, PERSIST_DIR, k=3)
    print(result)

    print("\n" + "=" * 70)
    print("Demo complete.")
    print("=" * 70)