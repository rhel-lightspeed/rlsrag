"""Test the download_model module."""

from unittest.mock import patch

from rlsrag.config import EMBED_MODEL, EMBED_MODEL_DIR
from rlsrag.download_model import main


def test_main():
    with patch("rlsrag.download_model.HuggingFaceEmbedding") as mock:
        main()
        mock.assert_called_once_with(
            model_name=EMBED_MODEL,
            cache_folder=EMBED_MODEL_DIR,
        )
