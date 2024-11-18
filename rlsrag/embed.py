"""Generate embeddings for the plaintext files."""

import logging
from glob import glob

from langchain.docstore.document import Document

from rlsrag import config
from rlsrag.infrastructure import get_vector_store

logger = logging.getLogger(__name__)


def read_chunks() -> list:
    """Read chunks from plaintext files for the embedding process.

    Returns:
        list: List of chunks to embed in the vector database.
    """
    plaintext_files = glob(f"{config.PLAINTEXT_DIR}/**/*.txt")
    chunks = [process_plaintext_file(file) for file in plaintext_files]

    # Flatten the list.
    return [x for xs in chunks for x in xs]


def process_plaintext_file(filename: str) -> list:
    """Process a plaintext file and split it into chunks.

    Args:
        filename (str): Path to the plaintext file.

    Returns:
        list: List of chunks from the plaintext file.
    """
    chunks = []

    # Get the file path without the plaintext directory prepended.
    file_path = filename.replace(config.PLAINTEXT_DIR, "").rsplit(".", 1)[0]

    # Loop over the chunks in the document and note the source document for each.
    with open(filename) as fileh:
        for piece in fileh.read().split("━" * 120):
            piece += f"\n\nSource document: {file_path}"
            chunks.append(Document(page_content=piece))

    return chunks


def batch_chunks(chunks: list, chunk_size: int = 2500) -> list:
    """Split the chunks into smaller chunks for embedding.

    Args:
        chunks (list): List of chunks to split.
        chunk_size (int, optional): Size of the chunks to split into. Defaults to 2500.

    Returns:
        list: List of smaller chunks.
    """
    return [
        chunks[i : i + chunk_size] for i in range(0, len(chunks), chunk_size)
    ]


def run_embed_pipeline() -> None:
    """Run the embedding pipeline."""
    logger.info("Running the embedding pipeline.")

    logger.info("Reading chunks from plaintext files.")
    chunks = read_chunks()
    vector_store = get_vector_store()

    logger.info(f"Embedding {len(chunks)} chunks.")
    for batch in batch_chunks(chunks):
        vector_store.add_documents(batch)

    logger.info("Finished embedding pipeline.")
    return None


if __name__ == "__main__":
    run_embed_pipeline()
