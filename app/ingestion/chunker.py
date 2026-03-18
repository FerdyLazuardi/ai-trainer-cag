"""
Token-aware text splitter using tiktoken.
Splits documents into chunks of ~800 tokens with 100-token overlap.
"""
import uuid
from dataclasses import dataclass, field

import tiktoken
from langchain_text_splitters import TokenTextSplitter

from app.config.settings import get_settings

settings = get_settings()

_ENCODING = "cl100k_base"


@dataclass
class TextChunk:
    """A single chunk ready for embedding and storage."""
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    text: str = ""
    chunk_index: int = 0
    token_count: int = 0
    metadata: dict = field(default_factory=dict)


def get_token_splitter(
    chunk_size: int = 800,
    chunk_overlap: int = 100,
) -> TokenTextSplitter:
    """Return a configured TokenTextSplitter."""
    return TokenTextSplitter(
        encoding_name=_ENCODING,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def count_tokens(text: str) -> int:
    """Count tokens in a string using tiktoken."""
    enc = tiktoken.get_encoding(_ENCODING)
    return len(enc.encode(text))


def chunk_text(
    text: str,
    metadata: dict | None = None,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
) -> list[TextChunk]:
    """
    Split raw text into token-aware chunks.

    Args:
        text: The document text to split.
        metadata: Optional metadata to attach to every chunk.
        chunk_size: Max tokens per chunk.
        chunk_overlap: Overlap tokens between consecutive chunks.

    Returns:
        List of TextChunk objects.
    """
    if not text.strip():
        return []

    splitter = get_token_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    raw_chunks = splitter.split_text(text)

    chunks: list[TextChunk] = []
    for idx, chunk_text_str in enumerate(raw_chunks):
        token_count = count_tokens(chunk_text_str)
        chunks.append(
            TextChunk(
                text=chunk_text_str,
                chunk_index=idx,
                token_count=token_count,
                metadata=metadata or {},
            )
        )

    return chunks
