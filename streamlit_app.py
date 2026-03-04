# streamlit_app.py

import streamlit as st
import time
import os

from app.data_generator import load_or_generate
from app.retrieval import index_data, hybrid_search
from app.config import MLFLOW_TRACKING_URI


# ─────────────────────────────────────────────────────────────
# Page Config (MUST be first Streamlit command)
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Semantic Transaction Search",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────
# MLflow Setup
# ─────────────────────────────────────────────────────────────
mlflow_enabled = False
try:
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("Transaction_Search")
    mlflow_enabled = True
except Exception:
    pass


# ─────────────────────────────────────────────────────────────
# Initialize Data (Cached)
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading data and building index... (first run takes ~2 min)")
def initialize():
    df = load_or_generate("data/synthetic_transactions.csv", 5000)
    index_data(df)
    return df

df = initialize()


# ─────────────────────────────────────────────────────────────
# Session State Initialization
# ─────────────────────────────────────────────────────────────
for key, default in [
    ("searched", False),
    ("results", []),
    ("filters", None),
    ("mlflow_run_id", None),
    ("latency_ms", None),
    ("k", 5),  # Default k
]:
    if key not in st.session_state:
        st.session_state[key] = default


def run_search(query: str):
    query = query.strip()
    if not query:
        return

    t0 = time.time()

    results, filters = hybrid_search(
        query,
        top_k=st.session_state.k
    )

    documents = results["documents"][0] if results.get("documents") else []
    distances = results["distances"][0] if results.get("distances") else []

    raw_results = []
    for i, (doc, dist) in enumerate(zip(documents, distances)):
        similarity = round((1 - dist) * 100, 2)
        raw_results.append({
            "rank": i + 1,
            "similarity": similarity,
            "description": doc
        })

    latency_ms = round((time.time() - t0) * 1000, 2)

    # ───────────────── MLflow Logging ─────────────────
    run_id = None
    if mlflow_enabled:
        try:
            from app.config import EMBEDDING_MODEL

            embedding_dim = 384
            manual_score = st.session_state.get("manual_score", None)

            with mlflow.start_run(run_name=f"search: {query[:40]}") as run:

                mlflow.log_param("query", query)
                mlflow.log_param("filters", str(filters))
                mlflow.log_param("k", st.session_state.k)
                mlflow.log_param("embedding_model", EMBEDDING_MODEL)
                mlflow.log_param("vector_db", "chromadb")
                mlflow.log_param("embedding_dim", embedding_dim)
                mlflow.log_param("num_transactions", len(df))

                mlflow.log_metric("latency_ms", latency_ms)
                mlflow.log_metric("num_results", len(raw_results))

                if raw_results:
                    avg_sim = sum(r["similarity"] for r in raw_results) / len(raw_results)
                    mlflow.log_metric("avg_similarity", round(avg_sim, 2))

                if manual_score is not None:
                    mlflow.log_metric("manual_relevance_score", manual_score)

                from app.retrieval import collection
                mlflow.log_metric("index_size", collection.count())

                run_id = run.info.run_id

        except Exception as e:
            print("MLflow logging error:", e)

    # ───────────────── Update Session State ─────────────────
    st.session_state.results = raw_results
    st.session_state.filters = filters
    st.session_state.mlflow_run_id = run_id
    st.session_state.latency_ms = latency_ms
    st.session_state.searched = True


def do_search():
    run_search(st.session_state.search_input)


# ─────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────
st.title("🔍 Semantic Transaction Search")
st.caption("Ask a financial question and see results ranked by semantic relevance.")

# Query input
st.text_input(
    "Search query",
    key="search_input",
    on_change=do_search,
    placeholder="e.g. high-value SWIFT transactions to India above 20000",
)

# 👇 K selector (max 15)
st.slider(
    "Number of results (k)",
    min_value=1,
    max_value=15,
    value=st.session_state.k,
    key="k"
)

col1, col2 = st.columns([1, 4])
with col1:
    if st.button("Search 🔍", use_container_width=True):
        do_search()


# ─────────────────────────────────────────────────────────────
# Results
# ─────────────────────────────────────────────────────────────
if st.session_state.searched:
    if st.session_state.results:

        status_parts = [
            f"⏱️ Latency: {st.session_state.latency_ms} ms",
            f"🔢 k = {st.session_state.k}"
        ]

        if st.session_state.mlflow_run_id:
            status_parts.append("📊 MLflow run logged")

        st.success("  |  ".join(status_parts))

        st.write("### Results")

        for r in st.session_state.results:
            with st.container():
                st.markdown(
                    f"**Rank #{r['rank']}** — **{r['similarity']}% similarity**"
                )
                st.info(r["description"])

    else:
        st.warning("No results found. Try a different query.")


# ─────────────────────────────────────────────────────────────
# Sidebar Example Queries
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("💡 Example Queries")

    examples = [
        "high value SWIFT transactions to India",
        "failed RTGS payments last 30 days",
        "corporate salary transfers above 20000",
        "pending transactions to Germany in January",
        "cross-border SME transfers returned",
        "low value domestic FEE transactions",
        "failed transactions with insufficient funds",
        "TRADE transfers to UAE above 15000",
    ]

    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state.search_input = ex
            run_search(ex)
            st.rerun()