"""API for querying the vector database."""

import logging

from fastapi import FastAPI
from pydantic import BaseModel

from rlsrag.embed import get_vector_store

logger = logging.getLogger(__name__)

app = FastAPI()

# Provisioning the vector store is expensive since the embedding model must be loaded
# into memory at this step. Only do it once per startup.
vector_store = get_vector_store()


class RagQuery(BaseModel):
    """Base model for making RAG queries."""

    query: str
    top_k: int = 5


@app.get("/")
def read_root() -> dict:
    return {"Hello": "World"}


@app.post("/query")
def update_item(query: RagQuery) -> dict:
    results = vector_store.similarity_search(query.query, k=query.top_k)
    return {"documents": results}
