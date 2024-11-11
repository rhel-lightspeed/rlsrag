"""Tests for the utils module."""


def test_update_content_with_metadata():
    from rlsrag.utils import update_content_with_metadata

    content = "This is a test."
    metadata = {
        "extra": {
            "document_kind": "test",
            "update_date": "2023-12-31",
        },
        "title": "Test",
        "path": "/test",
    }

    result = update_content_with_metadata(content, metadata)

    assert result.startswith("# Introduction")
    assert "This test document explains 'Test'" in result
    assert "https://access.redhat.com/test" in result
    assert "it was published on" in result

    metadata.pop("extra")
    result = update_content_with_metadata(content, metadata)
    assert result == content


def test_recently_updated():
    from rlsrag.utils import recently_updated

    metadata = {
        "extra": {
            "update_date": "2023-12-31",
        },
    }

    assert recently_updated(metadata)

    metadata = {
        "extra": {
            "update_date": "2016-12-31",
        },
    }

    assert not recently_updated(metadata)

    metadata = {
        "extra": {
            "update_date": "not a date",
        },
    }
    assert not recently_updated(metadata)


def test_process_markdown():
    from rlsrag.utils import process_markdown

    raw_markdown = """+++
title = "Test"
path = "/test"
extra = {document_kind = "test", update_date = "2023-12-31"}
+++
This is the first line of the content.
"""
    result = process_markdown(raw_markdown)
    assert "# Introduction" in result.text
    assert "This is the first line of the content." in result.text
    assert result.metadata["title"] == "Test"

    raw_markdown = """+++
+++
title = "Test"
+++
This is the first line of the content.
"""
    result = process_markdown(raw_markdown)
    assert result.text == ""
    assert result.metadata == {}

    raw_markdown = """+++
+++
title = "Test"
extra = {document_kind = "test", update_date = "1996-12-31"}
+++
This is the first line of the content.
"""
    result = process_markdown(raw_markdown)
    assert result.text == ""
    assert result.metadata == {}


def test_read_plaintext_files(tmp_path):
    from rlsrag.utils import read_plaintext_files

    test_doc = """+++
title = "Test"
path = "/test"
extra = {document_kind = "test", update_date = "2023-12-31"}
+++
This is the first line of the content.
"""

    # Use pytest's built in temporary file manager.
    d = tmp_path / "articles"
    d.mkdir()
    p = d / "00001.md"
    p.write_text(test_doc, encoding="utf-8")

    result = read_plaintext_files(d)
    assert len(result) == 1
    assert result[0].metadata["title"] == "Test"
    assert result[0].text == "This is the first line of the content."
