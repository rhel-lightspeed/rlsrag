"""Download the embedding model from Hugging Face."""

import logging

from rlsrag.embed import get_embedding_model

logger = logging.getLogger(__name__)


def main() -> None:
    """Download the embedding model from Hugging Face."""
    logger.info("Downloading the embedding model from Hugging Face.")
    get_embedding_model()
    return None


if __name__ == "__main__":
    main()  # pragma: no cover
