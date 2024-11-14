"""Configuration variables for the project."""

import os

# URI for connecting to the postgres database that holds the vectors.
POSTGRES_URI = "postgresql://postgres:secrete@127.0.0.1/vectorsandthings"

# Embedding model details.
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_MODEL_DIR = "./embedding_model"
EMBED_DIMENSION = 384
EMBED_COLLECTION_NAME = "rlsrag"

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
POSTGRES_CONNECTION = (
    f"postgresql+psycopg://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
)

# Documents to return from a RAG query (often called "top k").
RAG_TOP_K = 5
