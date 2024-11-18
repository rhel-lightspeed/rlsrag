"""API for querying the vector database."""

import logging

from fastapi import FastAPI
from pydantic import BaseModel

from rlsrag.infer import query_llm

logger = logging.getLogger(__name__)

app = FastAPI()


class RagQuery(BaseModel):
    """Base model for making RAG queries."""

    query: str


@app.get("/")
def read_root() -> dict:
    return {"Hello": "World"}


@app.post("/query")
def infer(query: RagQuery) -> dict:
    results = query_llm(query.query)
    return results
