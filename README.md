# 🔍 NLP Transaction Search Engine

A semantic search engine for financial transactions using sentence embeddings, ChromaDB, Streamlit UI, and MLflow experiment tracking.

## Project Structure

```
nlp-transaction-search/
├── app/
│   ├── __init__.py
│   ├── config.py          # Model name, paths, MLflow URI
│   ├── data_generator.py  # Synthetic data generation
│   ├── embedding.py       # Sentence-Transformers embeddings
│   ├── filters.py         # NLP → metadata filter extraction
│   └── retrieval.py       # ChromaDB indexing & hybrid search
├── streamlit_app.py       # Streamlit UI + MLflow logging
├── setup_and_run.py       # Local verification script
├── Dockerfile             # Python 3.10.11-slim image
├── docker-compose.yml     # App + MLflow services
├── requirements.txt
└── .dockerignore
```

---

## ▶️ Running Locally (VS Code / .venv)

### 1. Activate your virtual environment
```bash
# Windows
.venv\Scripts\activate

# Mac/Linux
source .venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Verify setup
```bash
python setup_and_run.py
```

### 4. Run the Streamlit app
```bash
streamlit run streamlit_app.py
```

Open browser at: **http://localhost:8501**

> **Note:** MLflow is optional locally. The app works without it — it will just skip logging if MLflow server isn't running.

### 5. (Optional) Run MLflow locally
```bash
mlflow server --host 0.0.0.0 --port 5000
```
Then open MLflow UI at: **http://localhost:5000**

---

## 🐳 Running with Docker

### Prerequisites
- Docker Desktop installed and running

### Build & Start
```bash
docker-compose up --build
```

### Access
| Service | URL |
|---------|-----|
| Streamlit App | http://localhost:8501 |
| MLflow UI | http://localhost:5000 |

### Stop
```bash
docker-compose down
```

### Rebuild from scratch (clear cache)
```bash
docker-compose down -v
docker-compose up --build
```

---

## 🔎 Example Queries

| Query | What it does |
|-------|-------------|
| `high value SWIFT transactions to India` | Filters cross-border, high value, India |
| `failed RTGS payments last 30 days` | Filters FAILED status, RTGS rail, date range |
| `corporate salary transfers above 20000` | Filters CORPORATE segment, SALARY type, amount |
| `pending transactions to Germany in January` | Filters PENDING, Germany, January date |
| `cross-border SME transfers returned` | Filters cross-border scope, SME segment, RETURNED |

---

## ⚙️ Configuration

Edit `app/config.py`:
```python
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # Model name
CHROMA_PATH = "./chroma_data"            # Vector DB path
```

The MLflow URI is set via environment variable `MLFLOW_TRACKING_URI` (defaults to `http://localhost:5000`).
