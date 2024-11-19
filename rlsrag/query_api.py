"""API for querying the vector database."""

import logging
from typing import Any

import orjson
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from rlsrag.infer import query_llm
from rlsrag.infrastructure import get_retriever

logger = logging.getLogger(__name__)

app = FastAPI()


# 💞 ️https://smhk.net/note/2023/09/fastapi-pretty-print-json/
class ORJSONPrettyResponse(JSONResponse):
    """Render a pretty printed JSON response."""

    def render(self, content: Any) -> bytes:
        """Render the response."""
        return orjson.dumps(
            content,
            option=orjson.OPT_NON_STR_KEYS
            | orjson.OPT_SERIALIZE_NUMPY
            | orjson.OPT_INDENT_2,
        )


class LLMInference(BaseModel):
    """Base model for performing LLM inference."""

    query: str


class RagQuery(BaseModel):
    """Base model for performing RAG query."""

    query: str
    top_k: int = 20
    score_threshold: float = 0.80


@app.get("/")
def read_root() -> dict:
    return {"Hello": "World"}


@app.post("/query", response_class=ORJSONPrettyResponse)
def query(query: RagQuery) -> dict:
    """Query the vector store for documents without doing any inference."""
    retriever = get_retriever(
        top_k=query.top_k, score_threshold=query.score_threshold
    )
    docs = retriever.invoke(query.query)
    result = [{"id": x.id, "content": x.page_content} for x in docs]
    return {"result": result}


@app.post("/infer", response_class=ORJSONPrettyResponse)
def infer(query: LLMInference) -> dict:
    """Perform LLM inference with WatsonX using RAG data."""
    results = query_llm(query.query)
    return results
