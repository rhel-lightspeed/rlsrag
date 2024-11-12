"""Tests for the embed functions."""

from unittest.mock import patch

from llama_index.embeddings.huggingface import HuggingFaceEmbedding


@patch("torch.cuda.is_available", return_value=False)
def test_get_embed_device(mock_torch):
    from rlsrag.embed import get_embed_device

    assert get_embed_device() == "cpu"


def test_get_embedding_model():
    from rlsrag.embed import get_embedding_model

    model = get_embedding_model()
    assert isinstance(model, HuggingFaceEmbedding)
