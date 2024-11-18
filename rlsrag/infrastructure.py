"""Infrastructure components for AI-related tasks."""

import logging

from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames
from langchain.retrievers.contextual_compression import (
    ContextualCompressionRetriever,
)
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


def get_retriever() -> ContextualCompressionRetriever:
    """Get a vector store for retrieval.

    This vector store instantiation is for querying the vector store. If you need to add
    embeddings to the vector store, use the get_vector_store() function.

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
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 20, "score_threshold": 0.95},
    )
    return ContextualCompressionRetriever(
        base_compressor=get_reranker(), base_retriever=retriever
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


def get_reranker() -> WatsonxRerank:
    """Get a reranker for use with the vector store.

    The reranker takes a list of documents coming out of the vector store and reranks
    them based on their vectors and the vectors from the user query.

    Returns:
        WatsonxRerank: WatsonX reranker object.
    """
    reranker_params = {
        "truncate_input_tokens": 512,
        "return_options": {"top_n": 2, "inputs": True, "query": True},
    }
    return WatsonxRerank(
        model_id="ibm/slate-125m-english-rtrvr-v2",
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
