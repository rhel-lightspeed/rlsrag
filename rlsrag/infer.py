"""Perform inference with an LLM."""

import logging

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

from rlsrag.infrastructure import get_chat_llm, get_retriever

logger = logging.getLogger(__name__)


def format_rag_docs(docs: list) -> str:
    """Return a list of RAG documents.

    Args:
        docs (list): Documents passed in via the RAG chain from the retriever.

    Returns:
        str: Formatted list of of docs.
    """
    # TODO: Maybe convert this to XML or JSON when we have metadata we want to use?
    return "\n".join(doc.page_content for doc in docs)


def generate_prompt() -> PromptTemplate:
    """Generate a prompt for the LLM.

    https://www.ibm.com/docs/en/watsonx/saas?topic=lab-sample-prompts#sample5b

    Returns:
        str: Prompt for use with the LLM for inference.
    """
    template = """
    Context:
    ###
    {context}
    ###

    You are a Linux and system administration assistant.
    Use the provided context between the ### lines above at a higher priority than any data container within the model.
    Always cite the source document in your response.
    If there is no good answer in the article, say "I don't know".
    You are managing Linux/Red Hat Enterprise Linux 9 server with a bash shell.
    Provide short responses in about 100 words, unless you are specifically asked for more details.
    If you need to store any data, assume it will be stored in the conversation.
    APPLY MARKDOWN formatting when possible.

    Question: {question}
    Answer: """
    return PromptTemplate.from_template(template)


def generate_chat_prompt() -> ChatPromptTemplate:
    """_summary_

    Returns:
        ChatPromptTemplate: _description_
    """
    system_prompt = """
    You are a Linux and system administration assistant.
    Use the provided context between the ### lines above at a higher priority than any data container within the model.
    Always cite the source document in your response.
    If there is no good answer in the article, say "I don't know".
    You are managing Linux/Red Hat Enterprise Linux 9 server with a bash shell.
    Provide short responses in about 100 words, unless you are specifically asked for more details.
    If you need to store any data, assume it will be stored in the conversation.
    APPLY MARKDOWN formatting when possible.

    {context}
    """
    prompt_fields = [("system", system_prompt.strip()), ("human", "{input}")]
    prompt = ChatPromptTemplate.from_messages(prompt_fields)
    return prompt


def clean_query(query: str) -> str:
    """Clean the user query to remove stop words.

    Args:
        query (str): The original user query.

    Returns:
        str: A modified query with stop words removed.
    """
    text_tokens = word_tokenize(query.lower())
    modified_tokens = [
        word for word in text_tokens if word not in stopwords.words()
    ]
    modified_query = TreebankWordDetokenizer().detokenize(modified_tokens)

    logger.info(f"🔧 Query modified: {query} >>> {modified_query}")

    return str(modified_query)


def query_llm(query: str) -> dict:
    """Query the LLM with a prompt, query, and RAG docs.

    Args:
        query (str): Query to use for inference.

    Returns:
        str: Response from the LLM.
    """
    retriever = get_retriever()
    llm = get_chat_llm()
    prompt = generate_chat_prompt()

    # Set up the RAG chain to fill in the prompt template, query the LLM, and parse the
    # output.
    # rag_chain = (
    #     {
    #         "context": retriever | format_rag_docs,
    #         "question": RunnablePassthrough(),
    #     }  # type: ignore[var-annotated]
    #     | prompt
    #     # | llm
    #     # | StrOutputParser()
    # )

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    return dict(rag_chain.invoke({"input": query}))
