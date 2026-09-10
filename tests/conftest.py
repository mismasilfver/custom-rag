import json
import zipfile
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Temporary data directory, cleaned up after each test."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def tmp_chroma_dir(tmp_path):
    """Temporary ChromaDB directory, cleaned up after each test."""
    chroma_dir = tmp_path / "chroma_db"
    return chroma_dir


@pytest.fixture
def sample_txt_file(tmp_path):
    """A small .txt file for testing uploads and indexing."""
    file_path = tmp_path / "sample.txt"
    file_path.write_text("This is a sample document for testing the RAG system.")
    return file_path


@pytest.fixture
def sample_pdf_file(tmp_path):
    """A dummy .pdf file (just bytes) for testing file management."""
    file_path = tmp_path / "sample.pdf"
    file_path.write_bytes(b"%PDF-1.4 fake pdf content for testing")
    return file_path


@pytest.fixture
def mock_ollama_responses():
    """Factory for mock Ollama HTTP responses."""

    def _make_response(status_code=200, body=None):
        mock_response = MagicMock()
        mock_response.status = status_code
        mock_response.read.return_value = json.dumps(body or {}).encode()
        return mock_response

    return _make_response


@pytest.fixture
def ollama_models_response():
    """Typical response from Ollama /api/tags endpoint."""
    return {
        "models": [
            {"name": "llama3.1:8b", "size": 4_700_000_000},
            {"name": "nomic-embed-text:latest", "size": 274_000_000},
            {"name": "mistral:7b", "size": 4_100_000_000},
        ]
    }


def _build_minimal_epub(file_path, title="Sample Book", content="Sample chapter text."):
    """Create a minimal valid EPUB container at ``file_path``.

    Uses only the standard library ``zipfile`` module so the fixture
    does not depend on ``ebooklib`` to create test data.
    """
    mimetype = b"application/epub+zip"
    ns_container = "urn:oasis:names:tc:opendocument:xmlns:container"
    container_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f'<container version="1.0" xmlns="{ns_container}">'
        "<rootfiles>"
        '<rootfile full-path="OEBPS/content.opf" '
        'media-type="application/oebps-package+xml"/>'
        "</rootfiles></container>"
    )
    chapter_xhtml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<!DOCTYPE html>"
        '<html xmlns="http://www.w3.org/1999/xhtml">'
        "<head><title>Chapter 1</title></head>"
        "<body>"
        f"<h1>{title}</h1>"
        f"<p>{content}</p>"
        "</body></html>"
    )
    content_opf = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<package version="3.0" xmlns="http://www.idpf.org/2007/opf">'
        "<metadata>"
        f'<dc:title xmlns:dc="http://purl.org/dc/elements/1.1/">{title}</dc:title>'
        "</metadata>"
        "<manifest>"
        '<item id="chapter1" href="chapter1.xhtml" '
        'media-type="application/xhtml+xml"/>'
        '<item id="toc" href="toc.ncx" '
        'media-type="application/x-dtbncx+xml"/>'
        "</manifest>"
        '<spine toc="toc"><itemref idref="chapter1"/></spine>'
        "</package>"
    )
    toc_ncx = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<ncx version="2005-1" xmlns="http://www.daisy.org/z3986/2005/ncx/">'
        '<head><meta name="dtb:uid" content="12345"/>'
        '<meta name="dtb:depth" content="1"/>'
        '<meta name="dtb:totalPageCount" content="0"/>'
        '<meta name="dtb:maxPageNumber" content="0"/></head>'
        "<docTitle><text>Sample Book</text></docTitle>"
        '<navMap><navPoint id="navpoint-1" playOrder="1">'
        "<navLabel><text>Chapter 1</text></navLabel>"
        '<content src="chapter1.xhtml"/></navPoint></navMap></ncx>'
    )

    with zipfile.ZipFile(file_path, "w") as zf:
        # EPUB spec requires mimetype to be first and uncompressed.
        zf.writestr("mimetype", mimetype, compress_type=zipfile.ZIP_STORED)
        zf.writestr("META-INF/container.xml", container_xml)
        zf.writestr("OEBPS/content.opf", content_opf)
        zf.writestr("OEBPS/chapter1.xhtml", chapter_xhtml)
        zf.writestr("OEBPS/toc.ncx", toc_ncx)


@pytest.fixture
def sample_epub_file(tmp_path):
    """A small valid .epub file for testing EPUB ingestion."""
    file_path = tmp_path / "sample.epub"
    _build_minimal_epub(
        file_path, content="This is sample chapter content for EPUB testing."
    )
    return file_path
