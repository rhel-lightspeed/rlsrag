"""Generate embeddings for the plaintext files."""

import logging

import tomllib
from langchain.docstore.document import Document
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from langchain_text_splitters import MarkdownHeaderTextSplitter

from rlsrag import config

logger = logging.getLogger(__name__)


def get_embedding_model() -> HuggingFaceEmbeddings:
    """Get the embedding model."""
    # Normalizing embeddings ensures that they're all the same size.
    return HuggingFaceEmbeddings(
        model_name=config.EMBED_MODEL,
        encode_kwargs={"batch_size": 128, "normalize_embeddings": True},
        cache_folder=config.EMBED_MODEL_DIR,
    )


def get_vector_store() -> PGVector:
    """Get the vector store."""
    return PGVector(
        embeddings=get_embedding_model(),
        embedding_length=config.EMBED_DIMENSION,
        collection_name=config.EMBED_COLLECTION_NAME,
        connection=config.POSTGRES_CONNECTION,
        logger=logger,
        use_jsonb=True,
    )


def load_documents() -> list:
    """Load the documents to embed."""
    logger.info("Loading documents.")
    text_loader_kwargs = {"autodetect_encoding": True}
    loader = DirectoryLoader(
        config.PLAINTEXT_DIR,
        glob="**/*.md",
        show_progress=True,
        loader_cls=TextLoader,
        loader_kwargs=text_loader_kwargs,
        use_multithreading=True,
    )
    return loader.load()


def split_document(doc: Document) -> list:
    """Split each document based on Markdown headers."""
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
    ]
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on, return_each_line=False)
    md_doc = md_splitter.split_text(doc.page_content)

    for i in range(len(md_doc)):
        md_doc[i].metadata = md_doc[i].metadata | doc.metadata
    return md_doc


def gather_chunks() -> list:
    """Gather the chunks to embed."""
    logger.info("Gathering documents.")

    # Read all of the documents.
    docs = load_documents()

    # Extract the TOML frontmatter from the documents.
    docs = extract_document_metadata(docs)

    # Split the documents into chunks using markdown headers.
    docs = [split_document(doc) for doc in docs]

    # Return a flat list of chunks.
    return [chunk for doc in docs for chunk in doc]


def extract_document_metadata(docs: list) -> list:
    """Extract the TOML frontmatter from each document."""
    for doc in docs:
        # Skip documents without TOML frontmatter.
        if not doc.page_content.startswith("+++"):
            continue

        # Split the metadata from the content.
        frontmatter, content = doc.page_content.split("\n+++\n", 1)
        extra_metadata = tomllib.loads(frontmatter.strip("+"))

        # Remove the "extra" dictionary.
        extra_metadata.pop("extra", None)

        # Replace the old content with just the content of the document.
        doc.page_content = content.strip("-").strip()

        # Update the metadata.
        doc.metadata["source"] = extra_metadata.get("path", None)
        doc.metadata = doc.metadata | extra_metadata

    return docs


def add_document_batches(doc_batch: list) -> None:
    """Helper function to handle adding document batches to the vector store."""
    get_vector_store().add_documents(doc_batch)
    return None


def run_embed_pipeline() -> None:
    """Run the embedding pipeline."""
    logger.info("Running the embedding pipeline.")
    chunks = gather_chunks()

    # Set batch size and divide documents into smaller batches
    batch_size = 128
    batches = [chunks[i : i + batch_size] for i in range(0, len(chunks), batch_size)]

    logger.info(f"Adding {len(chunks)} chunks in {len(batches)} batches to the vector store.")

    vector_store = get_vector_store()
    for counter, batch in enumerate(batches):
        vector_store.add_documents(batch)
        if (counter + 1) % 10 == 0:
            logger.info(f"Batches completed: {counter/len(batches)*100:.2f}%")

    logger.info("Finished embedding pipeline.")

    return None
