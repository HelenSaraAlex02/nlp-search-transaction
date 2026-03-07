# app/retrieval.py

import chromadb
from chromadb.config import Settings

from .embedding import embed_text
from .config import CHROMA_PATH
from .filters import extract_filters


# ============================================================
# Initialize Chroma Client
# (Chroma 0.3.29 → DuckDB backend)
# ============================================================

client = chromadb.Client(
    Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=CHROMA_PATH,
        anonymized_telemetry=False
    )
)

collection = client.get_or_create_collection(
    name="payments",
    metadata={"hnsw:space": "cosine"}
)


# ============================================================
# Utility
# ============================================================

def index_exists():
    """
    Check if embeddings already exist in the collection.
    Prevents rebuilding the index every Streamlit reload.
    """
    try:
        return collection.count() > 0
    except Exception:
        return False


# ============================================================
# Index Data
# ============================================================

def index_data(df):
    """
    Index transactions into ChromaDB.

    Runs ONLY if the collection is empty.
    Prevents rebuilding embeddings every app restart.
    """

    if index_exists():
        return

    documents = df["text_description"].tolist()
    ids = df["transaction_id"].astype(str).tolist()

    # Generate embeddings once
    embeddings = embed_text(documents)

    # Build metadata
    metadatas = [
        {
            "transaction_type": row["transaction_type"],
            "payment_rail": row["payment_rail"],
            "status": row["status"],
            "receiver_country": row["receiver_country"],
            "transaction_scope": row["transaction_scope"],
            "customer_segment": row["customer_segment"],
            "value_category": row["value_category"],
            "amount": float(row["amount"]),
            "currency": row["currency"],
            "date": int(row["date"].timestamp())
        }
        for _, row in df.iterrows()
    ]

    collection.add(
        documents=documents,
        ids=ids,
        embeddings=embeddings.tolist(),
        metadatas=metadatas
    )

    # Persist to disk
    client.persist()


# ============================================================
# Hybrid Search
# ============================================================

def hybrid_search(query, top_k=5):
    """
    Hybrid Search:
    - Metadata filtering
    - Semantic similarity search
    """

    # Safety clamp (avoid huge queries)
    top_k = max(1, min(int(top_k), 15))

    filters = extract_filters(query)

    query_embedding = embed_text([query])

    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=top_k,
        where=filters
    )

    return results, filters