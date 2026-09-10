"""Custom EPUB reader for LlamaIndex.

Extracts plain text from EPUB document items using EbookLib and BeautifulSoup.
"""

import logging
from typing import Any, Dict, List, Optional

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document

logger = logging.getLogger(__name__)


def _extract_first_heading(soup) -> Optional[str]:
    """Return text of the first h1/h2/h3 tag, or None."""
    for tag_name in ("h1", "h2", "h3"):
        heading = soup.find(tag_name)
        if heading:
            return heading.get_text(strip=True)
    return None


class EpubReader(BaseReader):
    """Read EPUB e-books and return plain-text LlamaIndex Documents."""

    def load_data(
        self,
        file,
        extra_info: Optional[Dict[str, Any]] = None,
        fs=None,
    ) -> List[Document]:
        """Extract text from ``file`` and return a single Document per EPUB.

        Args:
            file: Path to the EPUB file.
            extra_info: Optional metadata to attach to the Document.
            fs: fsspec filesystem (not supported; ignored with warning).

        Returns:
            A list containing one Document with the EPUB text, or an empty
            list if the file cannot be parsed.
        """
        if fs is not None:
            logger.warning(
                "fs was specified but EpubReader doesn't support loading "
                "from fsspec filesystems. Will load from local filesystem instead."
            )

        try:
            import ebooklib
            from bs4 import BeautifulSoup
            from ebooklib import epub
        except ImportError as err:
            raise ImportError(
                "EpubReader requires ebooklib and beautifulsoup4. "
                "Install them with: pip install ebooklib beautifulsoup4"
            ) from err

        book = epub.read_epub(str(file), options={"ignore_ncx": True})

        metadata = dict(extra_info) if extra_info else {}
        text_parts = []

        for item in book.get_items():
            if item.get_type() != ebooklib.ITEM_DOCUMENT:
                continue

            try:
                content = item.get_content().decode("utf-8", errors="ignore")
                soup = BeautifulSoup(content, "html.parser")

                # Remove non-content tags so they don't pollute the extracted text.
                for noise_tag in soup(["script", "style", "nav"]):
                    noise_tag.decompose()

                body_text = soup.get_text(separator="\n", strip=True)
                if not body_text:
                    continue

                chapter_title = getattr(item, "title", None) or _extract_first_heading(
                    soup
                )
                if chapter_title:
                    text_parts.append(f"# {chapter_title}\n\n{body_text}")
                else:
                    text_parts.append(body_text)
            except Exception as err:
                logger.warning(f"Skipping unreadable EPUB item in '{file}': {err}")
                continue

        if not text_parts:
            return []

        full_text = "\n\n".join(text_parts)
        return [Document(text=full_text, metadata=metadata)]
