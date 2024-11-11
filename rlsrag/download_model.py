"""Download the embedding model from Hugging Face."""

import logging

from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from rlsrag.config import EMBED_MODEL, EMBED_MODEL_DIR

logger = logging.getLogger(__name__)


def main() -> None:
    """Download the embedding model from Hugging Face."""
    logger.info("Downloading the embedding model from Hugging Face.")
    HuggingFaceEmbedding(
        model_name=EMBED_MODEL,
        cache_folder=EMBED_MODEL_DIR,
    )


if __name__ == "__main__":
    main()  # pragma: no cover
