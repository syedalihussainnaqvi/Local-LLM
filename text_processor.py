"""Text processing and smart sentence-aware chunking with rich metadata."""

import hashlib
import re
from typing import Any, Dict, List, Optional

import config


class TextProcessor:
    """Processes cleaned document text into structured, overlapping chunks respecting sentence boundaries."""

    def __init__(
        self,
        chunk_size: int = config.CHUNK_SIZE,
        chunk_overlap: int = config.CHUNK_OVERLAP,
    ):
        self.chunk_size = max(100, chunk_size)
        self.chunk_overlap = max(0, min(chunk_overlap, self.chunk_size // 2))

    @staticmethod
    def clean_text(raw_text: str) -> str:
        """Normalize line breaks and redundant spaces, trimming unnecessary fluff."""
        if not raw_text:
            return ""
        # Normalize carriage returns
        text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
        # Collapse multi-spaces within lines
        text = re.sub(r"[ \t]+", " ", text)
        # Collapse excessive newlines to double newlines
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def split_into_sentences_or_paragraphs(self, text: str) -> List[str]:
        """Split text along paragraph breaks or sentence terminals."""
        paragraphs = text.split("\n\n")
        segments = []

        for p in paragraphs:
            p = p.strip()
            if not p:
                continue

            # If paragraph fits within chunk_size, keep as single segment
            if len(p) <= self.chunk_size:
                segments.append(p)
                continue

            # Otherwise, split paragraph by sentence delimiters
            # Matches ., !, ? followed by whitespace or end of line
            sentence_splits = re.split(r"(?<=[.!?])\s+", p)
            current_sentence = ""

            for s in sentence_splits:
                s = s.strip()
                if not s:
                    continue
                if len(s) > self.chunk_size:
                    # Very long sentence without punctuation: split by commas or words
                    subparts = re.split(r"(?<=[,;])\s+", s)
                    for sub in subparts:
                        sub = sub.strip()
                        if sub:
                            segments.append(sub)
                else:
                    segments.append(s)

        return segments

    def chunk_document(
        self,
        text: str,
        source_url: str,
        title: str = "",
        retrieved_at: str = "",
        doc_hash: str = "",
    ) -> List[Dict[str, Any]]:
        """
        Produce a list of metadata-rich chunks from the input document.
        """
        cleaned = self.clean_text(text)
        if not cleaned:
            return []

        if not doc_hash:
            doc_hash = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()

        segments = self.split_into_sentences_or_paragraphs(cleaned)
        chunks = []
        current_chunk_segments: List[str] = []
        current_length = 0
        current_section = ""
        chunk_idx = 0

        for seg in segments:
            # Check if this segment is a markdown heading
            if seg.startswith("#") or seg.startswith("## "):
                current_section = seg.lstrip("# ").strip()

            seg_len = len(seg)

            if current_length + seg_len > self.chunk_size and current_chunk_segments:
                # Emit accumulated chunk
                chunk_text = " ".join(current_chunk_segments).strip()
                if chunk_text:
                    chunk_id = f"{doc_hash[:12]}_{chunk_idx:04d}"
                    chunk_hash = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()
                    chunks.append({
                        "chunk_id": chunk_id,
                        "source_url": source_url,
                        "title": title or source_url,
                        "section": current_section,
                        "content": chunk_text,
                        "chunk_hash": chunk_hash,
                        "retrieved_at": retrieved_at,
                        "char_count": len(chunk_text),
                    })
                    chunk_idx += 1

                # Retain overlap from end of current segments
                overlap_segments = []
                overlap_len = 0
                for prev_seg in reversed(current_chunk_segments):
                    if overlap_len + len(prev_seg) <= self.chunk_overlap:
                        overlap_segments.insert(0, prev_seg)
                        overlap_len += len(prev_seg)
                    else:
                        break

                current_chunk_segments = overlap_segments + [seg]
                current_length = sum(len(s) for s in current_chunk_segments)
            else:
                current_chunk_segments.append(seg)
                current_length += seg_len

        # Final trailing chunk
        if current_chunk_segments:
            chunk_text = " ".join(current_chunk_segments).strip()
            if chunk_text:
                chunk_id = f"{doc_hash[:12]}_{chunk_idx:04d}"
                chunk_hash = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()
                chunks.append({
                    "chunk_id": chunk_id,
                    "source_url": source_url,
                    "title": title or source_url,
                    "section": current_section,
                    "content": chunk_text,
                    "chunk_hash": chunk_hash,
                    "retrieved_at": retrieved_at,
                    "char_count": len(chunk_text),
                })

        return chunks


def process_and_chunk(
    text: str,
    source_url: str,
    title: str = "",
    retrieved_at: str = "",
    doc_hash: str = "",
) -> List[Dict[str, Any]]:
    processor = TextProcessor()
    return processor.chunk_document(
        text=text,
        source_url=source_url,
        title=title,
        retrieved_at=retrieved_at,
        doc_hash=doc_hash,
    )
