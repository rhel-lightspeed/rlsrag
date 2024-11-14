"""Test the download_model module."""

from unittest.mock import patch

from rlsrag.download_model import main


def test_main():
    with patch("rlsrag.download_model.get_embedding_model") as mock:
        main()
        mock.assert_called_once()
