"""Configuration variables for the project."""

import os

from pydantic import SecretStr

# URI for connecting to the postgres database that holds the vectors.
POSTGRES_URI = "postgresql://postgres:secrete@127.0.0.1/vectorsandthings"

# Embedding model details.
EMBED_MODEL = "ibm/slate-125m-english-rtrvr-v2"
EMBED_COLLECTION_NAME = "rlsrag"

# LLM model.
LLM_MODEL = "ibm/granite-3-2b-instruct"

# IBM WatsonX details.
WATSONX_PROJECT_ID = os.environ.get("WATSONX_PROJECT_ID", None)
WATSONX_APIKEY = os.environ.get("WATSONX_APIKEY", None)
WATSONX_URL = SecretStr(
    os.environ.get("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
)

# Increasing the embedding batch size can speed up RAG training but
# it may need to be lowered on GPUs with less than 24GB RAM.
EMBED_BATCH_SIZE = 64

# Location of the plaintext files to use for RAG training.
PLAINTEXT_DIR = os.environ.get("PLAINTEXT_DIR", "./plaintext")

# Postgres vector store credentials.
# Set these via environment variables in production.
POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "127.0.0.1")
POSTGRES_PORT = os.environ.get("POSTGRES_PORT", "5432")
POSTGRES_USER = os.environ.get("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "secrete")
POSTGRES_DB = os.environ.get("POSTGRES_DB", "vectorsandthings")
POSTGRES_TABLE = os.environ.get("POSTGRES_TABLE", "vectors")
POSTGRES_CONNECTION = f"postgresql+psycopg://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

# Vector store collection name.
VECTOR_COLLECTION = "rlsrag"

# Documents to return from a RAG query (often called "top k").
RAG_TOP_K = 5

# Config checks.
if not WATSONX_PROJECT_ID:
    raise ValueError("WATSONX_PROJECT_ID")
if not WATSONX_APIKEY:
    raise ValueError("WATSONX_APIKEY")
