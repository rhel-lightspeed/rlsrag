"""Configuration variables for the project."""

# URI for connecting to the postgres database that holds the vectors.
POSTGRES_URI = "postgresql://postgres:secrete@127.0.0.1/vectorsandthings"

# Embedding model details.
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
EMBED_MODEL_DIR = "./embedding_model"

# Increasing the embedding batch size can speed up RAG training but
# it may need to be lowered on GPUs with less than 24GB RAM.
EMBED_BATCH_SIZE = 512

# Location of the plaintext files to use for RAG training.
PLAINTEXT_DIR = "./plaintext"
