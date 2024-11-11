"""Utilities for RAG training."""

import logging
from datetime import datetime
from glob import glob

import dateparser
import tomllib
from llama_index.core import Document

logger = logging.getLogger(__name__)


def update_content_with_metadata(content: str, metadata: dict) -> str:
    """Add metadata to the content."""
    if "extra" not in metadata:
        return content

    extra_content = (
        "# Introduction\n\n"
        f"This {metadata['extra']['document_kind']} document explains '{metadata['title']}' and "
        f"it is available online at https://access.redhat.com{metadata['path']} and "
        f"it was published on {metadata['extra']['update_date']}.\n\n"
    )
    return extra_content + content


def recently_updated(metadata: dict, years_old: int = 5) -> bool:
    """Check if a document has been updated recently."""
    current_year = datetime.now().year
    parsed_update_date = dateparser.parse(metadata["extra"]["update_date"])

    # Let's stop here if we can't parse the data for some reason.
    if not parsed_update_date:
        return False

    return parsed_update_date.year >= current_year - years_old


def process_markdown(raw_markdown: str) -> Document:
    """Process a markdown file that might have TOML frontmatter at the top."""
    # Our markdown has TOML-formatted frontmatter at the top.
    frontmatter, content = raw_markdown.split("\n+++\n", 1)

    # A lot of the content has a leading "---" that we need to strip.
    content = content.strip("-").strip()

    # Load the TOML metadata.
    metadata = tomllib.loads(frontmatter.strip("+").strip())

    # Skip this document if it is missing an 'extra' key.
    # Skip this document if it was not updated recently.
    if "extra" not in metadata or not recently_updated(metadata):
        return Document()

    # Add some basic data from the metadata to the page content.
    text = update_content_with_metadata(content, metadata)

    # Assemble the document object and append it to the list.
    return Document(text=text, metadata=metadata)  # type: ignore[call-arg]


def read_plaintext_files(plaintext_dir: str) -> list:
    """Read all the plain text files in the directory."""
    logging.info("Reading plain text files...")
    plaintext_files = glob(f"{plaintext_dir}/**/[0-9]*.md", recursive=True)

    markdown_docs = []
    for counter, plaintext_file in enumerate(plaintext_files):
        with open(plaintext_file) as fileh:
            parsed_document = process_markdown(fileh.read())
            if parsed_document:
                markdown_docs.append(parsed_document)

        if (counter + 1) % 5000 == 0:
            logging.info(f"Processed {counter + 1} documents.")

    logging.info(f"Loaded {len(markdown_docs)} documents.")
    logging.info(f"Excluded {len(plaintext_files) - len(markdown_docs)} documents.")

    return markdown_docs
