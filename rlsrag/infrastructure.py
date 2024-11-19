"""Infrastructure components for AI-related tasks."""

import logging

from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames
from langchain.retrievers.contextual_compression import (
    ContextualCompressionRetriever,
)
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain_ibm import (
    ChatWatsonx,
    WatsonxEmbeddings,
    WatsonxLLM,
    WatsonxRerank,
)
from langchain_postgres import PGVector

from rlsrag import config

logger = logging.getLogger(__name__)


def get_embedding_model() -> WatsonxEmbeddings:
    """Get the embedding model for use with searches and embeddings.

    Models: https://www.ibm.com/products/watsonx-ai/foundation-models
    Params: https://ibm.github.io/watsonx-ai-python-sdk/fm_embeddings.html

    Returns:
        WatsonxEmbeddings: Embedding model object.
    """
    return WatsonxEmbeddings(
        model_id=config.EMBED_MODEL,
        url=config.WATSONX_URL,
        project_id=config.WATSONX_PROJECT_ID,
        params={
            EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 512,
            EmbedTextParamsMetaNames.RETURN_OPTIONS: {"input_text": True},
        },
    )


def get_retriever(
    top_k: int = 20, score_threshold: float = 0.80
) -> VectorStoreRetriever:
    """Get a vector store for retrieval.

    This vector store instantiation is for querying the vector store. If you need to add
    embeddings to the vector store, use the get_vector_store() function.

    Args:
        top_k (int, optional): Number of documents to return. Defaults to 20.
        score_threshold (float, optional): Threshold for document relevance. Defaults to
            0.90.

    Returns:
        PGVector: Returns a vector store object.
    """
    vector_store = PGVector(
        embeddings=get_embedding_model(),
        collection_name=config.VECTOR_COLLECTION,
        connection=config.POSTGRES_CONNECTION,
        logger=logger,
        use_jsonb=True,
    )
    return vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": top_k, "score_threshold": score_threshold},
    )


def get_retriever_with_reranker() -> ContextualCompressionRetriever:
    """Get a retriever with a reranker attached.

    This function gives you a retriever with a reranker attached that will rerank
    documents based on their vectors and the vectors from the user query.

    Returns:
        ContextualCompressionRetriever: Retriever with reranker attached.
    """
    return ContextualCompressionRetriever(
        base_compressor=get_reranker(), base_retriever=get_retriever()
    )


def get_vector_store() -> PGVector:
    """Generate a vector store for embedding.

    This vector store instantiation is for generating embeddings and storing them in
    postgres. If you simply need to query the vector store, use the get_retriever()
    function.

    Returns:
        PGVector: Returns a vector store object.
    """
    return PGVector(
        embeddings=get_embedding_model(),
        collection_name=config.VECTOR_COLLECTION,
        connection=config.POSTGRES_CONNECTION,
        use_jsonb=True,
        pre_delete_collection=True,
    )


def get_reranker(top_n: int = 2) -> WatsonxRerank:
    """Get a reranker for use with the vector store.

    The reranker takes a list of documents coming out of the vector store and reranks
    them based on their vectors and the vectors from the user query.

    Args:
        top_n (int, optional): Number of documents to return. Defaults to 2.

    Returns:
        WatsonxRerank: WatsonX reranker object.
    """
    reranker_params = {
        "truncate_input_tokens": 512,
        "return_options": {"top_n": top_n, "inputs": True, "query": True},
    }
    return WatsonxRerank(
        model_id=config.EMBED_MODEL,
        url=config.WATSONX_URL,
        project_id=config.WATSONX_PROJECT_ID,
        params=reranker_params,
    )


def get_llm() -> WatsonxLLM:
    """Get a language model for use with the vector store.

    The language model takes a list of documents coming out of the vector store and reranks
    them based on their vectors and the vectors from the user query.

    Returns:
        WatsonxLLM: WatsonX reranker object.
    """
    llm_params = {
        "decoding_method": "greedy",
        "max_new_tokens": 2048,
        "min_new_tokens": 0,
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 1.0,
        "repetition_penalty": 1.1,
        "return_options": {
            "input_tokens": True,
            "generated_tokens": True,
            "token_logprobs": True,
            "token_ranks": True,
        },
    }
    return WatsonxLLM(
        model_id=config.LLM_MODEL,
        url=config.WATSONX_URL,
        project_id=config.WATSONX_PROJECT_ID,
        params=llm_params,
    )


def get_chat_llm() -> ChatWatsonx:
    """_summary_

    Returns:
        ChatWatsonX: _description_
    """
    llm_params = {
        "decoding_method": "greedy",
        "max_new_tokens": 2048,
        "min_new_tokens": 0,
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 1.0,
        "repetition_penalty": 1.1,
        "return_options": {
            "input_tokens": True,
            "generated_tokens": True,
            "token_logprobs": True,
            "token_ranks": True,
        },
    }
    return ChatWatsonx(
        model_id=config.LLM_MODEL,
        url=config.WATSONX_URL,
        project_id=config.WATSONX_PROJECT_ID,
        params=llm_params,
    )
