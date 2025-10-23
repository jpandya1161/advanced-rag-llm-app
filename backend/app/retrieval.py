from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List, Sequence, Set

from .embeddings import HashingEmbedder, cosine, tokenize
from .models import DocumentChunk, RetrievalResult


class HybridRetriever:
    def __init__(self, embedder: HashingEmbedder) -> None:
        self.embedder = embedder

    def search(self, query: str, chunks: Sequence[DocumentChunk], top_k: int) -> List[RetrievalResult]:
        if not chunks:
            return []

        query_vector = self.embedder.embed(query)
        query_terms = set(tokenize(query))
        keyword_scores = self._bm25_scores(query_terms, chunks)
        vector_scores = {
            chunk.chunk_id: max(0.0, cosine(query_vector, chunk.embedding)) for chunk in chunks
        }

        vector_ranks = self._rank_map(vector_scores)
        keyword_ranks = self._rank_map(keyword_scores)
        results: List[RetrievalResult] = []
        for chunk in chunks:
            vector_rank = vector_ranks.get(chunk.chunk_id, len(chunks) + 1)
            keyword_rank = keyword_ranks.get(chunk.chunk_id, len(chunks) + 1)
            rrf = (1.0 / (60 + vector_rank)) + (1.0 / (60 + keyword_rank))
            overlap = self._lexical_overlap(query_terms, set(tokenize(chunk.text)))
            score = rrf + (0.15 * overlap)
            results.append(
                RetrievalResult(
                    chunk=chunk,
                    score=score,
                    vector_score=vector_scores[chunk.chunk_id],
                    keyword_score=keyword_scores[chunk.chunk_id],
                    lexical_overlap=overlap,
                    rank=0,
                )
            )

        results.sort(key=lambda result: result.score, reverse=True)
        for index, result in enumerate(results, start=1):
            result.rank = index
        return results[:top_k]

    @staticmethod
    def _rank_map(scores: Dict[str, float]) -> Dict[str, int]:
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return {chunk_id: index for index, (chunk_id, _) in enumerate(ranked, start=1)}

    @staticmethod
    def _lexical_overlap(query_terms: Set[str], chunk_terms: Set[str]) -> float:
        if not query_terms:
            return 0.0
        return len(query_terms & chunk_terms) / len(query_terms)

    @staticmethod
    def _bm25_scores(query_terms: Set[str], chunks: Sequence[DocumentChunk]) -> Dict[str, float]:
        if not query_terms:
            return {chunk.chunk_id: 0.0 for chunk in chunks}

        tokenized = {chunk.chunk_id: tokenize(chunk.text) for chunk in chunks}
        doc_lengths = {chunk_id: len(tokens) for chunk_id, tokens in tokenized.items()}
        avgdl = sum(doc_lengths.values()) / max(len(doc_lengths), 1)
        document_frequency: Counter[str] = Counter()
        for tokens in tokenized.values():
            document_frequency.update(set(tokens))

        scores: Dict[str, float] = {}
        total_docs = len(chunks)
        k1 = 1.5
        b = 0.75
        for chunk in chunks:
            tokens = tokenized[chunk.chunk_id]
            counts = Counter(tokens)
            score = 0.0
            for term in query_terms:
                if counts[term] == 0:
                    continue
                df = document_frequency[term]
                idf = math.log(1 + ((total_docs - df + 0.5) / (df + 0.5)))
                numerator = counts[term] * (k1 + 1)
                denominator = counts[term] + k1 * (1 - b + b * (doc_lengths[chunk.chunk_id] / avgdl))
                score += idf * (numerator / denominator)
            scores[chunk.chunk_id] = score
        return scores
