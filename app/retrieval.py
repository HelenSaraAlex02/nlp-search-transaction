# app/retrieval.py

import chromadb
from chromadb.config import Settings
from .embedding import embed_text
from .config import CHROMA_PATH
from .filters import extract_filters

# ============================================================
# Initialize Chroma Client (0.3.29 API - uses DuckDB, no SQLite issue)
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


def is_collection_empty():
    return collection.count() == 0


def index_data(df):
    """Index transactions into ChromaDB. Runs only if collection is empty."""

    if not is_collection_empty():
        return

    documents = df["text_description"].tolist()
    ids = df["transaction_id"].astype(str).tolist()
    embeddings = embed_text(documents)

    metadatas = []
    for _, row in df.iterrows():
        metadatas.append({
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
        })

    collection.add(
        documents=documents,
        ids=ids,
        embeddings=embeddings.tolist(),
        metadatas=metadatas
    )

    client.persist()  # Required in 0.3.x to save to disk


def hybrid_search(query, top_k=5):
    """Hybrid search: metadata filters + semantic ANN."""

    # Safety clamp (UI protection)
    top_k = max(1, min(int(top_k), 15))

    filters = extract_filters(query)
    query_embedding = embed_text([query])

    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=top_k,
        where=filters
    )

    return results, filters