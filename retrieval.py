"""Hybrid semantic and lexical retrieval engine with metadata tracking."""

import logging
import math
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import config
from embeddings import LocalTextEmbedder, get_embedder
from storage import StorageManager, get_storage

logger = logging.getLogger("retrieval")


class HybridRetriever:
    """
    Hybrid retriever combining dense vector cosine similarity and BM25 term weighting.
    Caches vectors for stored chunks and dynamically re-indexes when chunks change.
    """

    def __init__(self, storage: Optional[StorageManager] = None, embedder: Optional[LocalTextEmbedder] = None):
        self.storage = storage or get_storage()
        self.embedder = embedder or get_embedder()
        self._cached_chunks: List[Dict[str, Any]] = []
        self._chunk_vectors: Optional[np.ndarray] = None
        self._doc_lens: List[int] = []
        self._avg_doc_len: float = 1.0
        self._df: Dict[str, int] = {}
        self._total_docs: int = 0
        self._last_index_time: float = 0

    def refresh_index(self) -> None:
        """Fetch all active chunks from storage and build in-memory vector index."""
        chunks = self.storage.get_all_chunks()
        self._cached_chunks = chunks
        self._total_docs = len(chunks)

        if not chunks:
            self._chunk_vectors = np.empty((0, self.embedder.vector_dim), dtype=np.float32)
            self._doc_lens = []
            self._avg_doc_len = 1.0
            self._df = {}
            return

        texts = [c["content"] for c in chunks]
        self._chunk_vectors = self.embedder.embed_batch(texts)

        # Build BM25 statistics
        self._df = {}
        self._doc_lens = []
        total_len = 0

        for t in texts:
            words = set(re.findall(r"\b[a-zA-Z0-9_]{2,}\b", t.lower()))
            self._doc_lens.append(len(words))
            total_len += len(words)
            for w in words:
                self._df[w] = self._df.get(w, 0) + 1

        self._avg_doc_len = max(1.0, total_len / max(1, self._total_docs))
        logger.debug(f"Retriever indexed {len(chunks)} chunks.")

    def _bm25_score(self, query_terms: List[str], chunk_idx: int, k1: float = 1.5, b: float = 0.75) -> float:
        """Calculate BM25 relevance score for a given chunk index."""
        if chunk_idx >= len(self._cached_chunks):
            return 0.0

        content = self._cached_chunks[chunk_idx]["content"].lower()
        chunk_words = re.findall(r"\b[a-zA-Z0-9_]{2,}\b", content)
        if not chunk_words:
            return 0.0

        tf_map = {}
        for w in chunk_words:
            tf_map[w] = tf_map.get(w, 0) + 1

        doc_len = self._doc_lens[chunk_idx] if chunk_idx < len(self._doc_lens) else len(chunk_words)
        score = 0.0

        for q in query_terms:
            tf = tf_map.get(q, 0)
            if tf == 0:
                continue

            df = self._df.get(q, 1)
            # Standard Lucene/BM25 IDF formula
            idf = math.log(1.0 + (self._total_docs - df + 0.5) / (df + 0.5))
            if idf < 0:
                idf = 0.0

            numerator = tf * (k1 + 1.0)
            denominator = tf + k1 * (1.0 - b + b * (doc_len / self._avg_doc_len))
            score += idf * (numerator / denominator)

        return score

    def search(
        self,
        query: str,
        top_k: int = config.RETRIEVAL_TOP_K,
        threshold: float = config.SIMILARITY_THRESHOLD,
        allowed_urls: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search indexed chunks using hybrid scoring (cosine similarity + BM25).
        Returns top-k chunks with metadata and normalized relevance score.
        If allowed_urls is provided, limits search to chunks from those URLs.
        """
        # Always check and refresh index if chunk count changed
        all_chunks = self.storage.get_all_chunks()
        if len(all_chunks) != len(self._cached_chunks):
            self.refresh_index()

        if not self._cached_chunks or self._chunk_vectors is None or len(self._chunk_vectors) == 0:
            return []

        clean_query = query.strip()
        if not clean_query:
            return []

        allowed_set = set(allowed_urls) if allowed_urls else None

        query_terms = re.findall(r"\b[a-zA-Z0-9_]{2,}\b", clean_query.lower())
        query_vec = self.embedder.embed_text(clean_query)

        # Dense vector cosine similarity
        cosine_scores = np.dot(self._chunk_vectors, query_vec)

        # Combine with BM25 scores
        scored_candidates: List[Tuple[float, Dict[str, Any]]] = []

        max_bm25 = 1.0
        bm25_scores = [self._bm25_score(query_terms, idx) for idx in range(len(self._cached_chunks))]
        if bm25_scores:
            max_bm25 = max(max(bm25_scores), 1.0)

        for idx, chunk in enumerate(self._cached_chunks):
            # If allowed_urls specified, filter strictly to allowed URLs
            if allowed_set and chunk.get("source_url") not in allowed_set:
                continue

            cos_s = float(cosine_scores[idx])
            bm_s = float(bm25_scores[idx]) / max_bm25  # normalize BM25 to ~ [0, 1]

            # Hybrid score: 60% semantic similarity + 40% keyword match
            hybrid_score = round(0.6 * max(0.0, cos_s) + 0.4 * bm_s, 4)

            if hybrid_score >= threshold:
                result_chunk = dict(chunk)
                result_chunk["score"] = hybrid_score
                scored_candidates.append((hybrid_score, result_chunk))

        # Sort descending by score
        scored_candidates.sort(key=lambda x: x[0], reverse=True)

        # Dynamic filtering: filter out weak trailing chunks that fall below 50% of the top score
        if scored_candidates:
            top_score = scored_candidates[0][0]
            cutoff = max(threshold, 0.28, 0.50 * top_score)
            scored_candidates = [sc for sc in scored_candidates if sc[0] >= cutoff]

        # Deduplicate near-identical snippets
        selected = []
        seen_hashes = set()

        for score, ch in scored_candidates:
            h = ch.get("chunk_hash")
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            selected.append(ch)
            if len(selected) >= top_k:
                break

        return selected


def format_context_for_prompt(chunks: List[Dict[str, Any]], max_chars: int = config.MAX_CONTEXT_LENGTH) -> Tuple[str, List[Dict[str, str]]]:
    """
    Format retrieved chunks into a clear, trustworthy context block.
    Returns:
        (formatted_text: str, source_citations: List[Dict[str, str]])
    """
    if not chunks:
        return "", []

    context_parts = []
    citations = []
    seen_urls = set()
    total_len = 0

    for i, ch in enumerate(chunks, 1):
        url = ch.get("source_url", "")
        title = ch.get("title", url)
        section = ch.get("section", "")
        content = ch.get("content", "").strip()
        retrieved_at = ch.get("retrieved_at", "")

        header = f"[Source {i}: {title}] ({url})"
        if section:
            header += f" - Section: {section}"

        snippet = f"{header}\n{content}"
        snippet_len = len(snippet)

        if total_len + snippet_len > max_chars and context_parts:
            # Reached max context length budget
            break

        context_parts.append(snippet)
        total_len += snippet_len

        if url and url not in seen_urls:
            seen_urls.add(url)
            citations.append({
                "url": url,
                "title": title,
                "retrieved_at": retrieved_at,
                "score": ch.get("score", 0.0),
            })

    formatted_context = "\n\n".join(context_parts)
    return formatted_context, citations


# Singleton accessor
_GLOBAL_RETRIEVER: Optional[HybridRetriever] = None

def get_retriever() -> HybridRetriever:
    global _GLOBAL_RETRIEVER
    if _GLOBAL_RETRIEVER is None:
        _GLOBAL_RETRIEVER = HybridRetriever()
        _GLOBAL_RETRIEVER.refresh_index()
    return _GLOBAL_RETRIEVER
