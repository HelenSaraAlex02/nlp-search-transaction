# app/config.py
import os
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHROMA_PATH = "./chroma_data"

# MLflow: use environment variable with fallback to localhost for local dev

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
DATA_PATH = os.path.join("data", "synthetic_transactions.csv")
