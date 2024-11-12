"""Generate embeddings for the plaintext files."""

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from torch.cuda import is_available as is_gpu_available

from rlsrag.config import EMBED_BATCH_SIZE, EMBED_MODEL, EMBED_MODEL_DIR


def get_embed_device() -> str:
    """Check if a GPU is available."""
    return "cuda" if is_gpu_available() else "cpu"


def get_embedding_model() -> HuggingFaceEmbedding:
    """Get the embedding model."""
    embed_model = HuggingFaceEmbedding(
        model_name=EMBED_MODEL,
        cache_folder=EMBED_MODEL_DIR,
        device=get_embed_device(),
        embed_batch_size=EMBED_BATCH_SIZE,
        trust_remote_code=True,
    )
    return embed_model
