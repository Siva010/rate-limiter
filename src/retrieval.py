# src/retrieval.py
from __future__ import annotations

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from typing import List, Tuple
from config import (
    EMBEDDING_MODEL, PINECONE_API_KEY, PINECONE_INDEX_NAME
)

# Module-level model cache to avoid reloading on every instantiation
_model_cache: dict[str, SentenceTransformer] = {}


def _get_model(model_name: str) -> SentenceTransformer:
    """Return a cached SentenceTransformer model."""
    if model_name not in _model_cache:
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


class BM25Retriever:
    def __init__(self, documents: List[dict]):
        self.documents = documents
        self.tokenized = [doc["text"].split() for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized)

    def retrieve(self, query: str, k: int) -> List[Tuple[dict, float]]:
        scores = self.bm25.get_scores(query.split())
        ranked = sorted(zip(self.documents, scores), key=lambda x: x[1], reverse=True)
        return ranked[:k]


class PineconeDenseRetriever:
    def __init__(self):
        self.model = _get_model(EMBEDDING_MODEL)
        pc = Pinecone(api_key=PINECONE_API_KEY)
        self.index = pc.Index(PINECONE_INDEX_NAME)

    def retrieve(self, query: str, k: int) -> List[Tuple[dict, float]]:
        q_emb = self.model.encode(query).tolist()
        res = self.index.query(vector=q_emb, top_k=k, include_metadata=True)
        return [
            ({"id": match["id"], "text": match["metadata"]["text"]}, match["score"])
            for match in res["matches"]
        ]


class DenseRetriever:
    """Local dense retriever using sentence embeddings (for CLI/testing without Pinecone)."""
    def __init__(self, documents: List[dict], model_name: str):
        self.documents = documents
        self.model = _get_model(model_name)
        self.embeddings = self.model.encode(
            [doc["text"] for doc in documents], show_progress_bar=False
        )

    def retrieve(self, query: str, k: int) -> List[Tuple[dict, float]]:
        q_emb = self.model.encode(query)
        # Cosine similarity
        norms = np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(q_emb)
        scores = np.dot(self.embeddings, q_emb) / np.where(norms == 0, 1, norms)
        ranked_idx = np.argsort(scores)[::-1][:k]
        return [(self.documents[i], float(scores[i])) for i in ranked_idx]


class HybridRetriever:
    def __init__(self, sparse: BM25Retriever, dense, alpha: float):
        self.sparse = sparse
        self.dense = dense
        self.alpha = alpha
        self.doc_map = {doc["id"]: doc for doc in sparse.documents}

    def retrieve(self, query: str, k: int) -> List[dict]:
        scores: dict[str, float] = {}

        for doc, s in self.sparse.retrieve(query, k):
            scores[doc["id"]] = scores.get(doc["id"], 0.0) + (1 - self.alpha) * s

        for doc, s in self.dense.retrieve(query, k):
            scores[doc["id"]] = scores.get(doc["id"], 0.0) + self.alpha * s
            self.doc_map.setdefault(doc["id"], doc)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        return [self.doc_map[i] for i, _ in ranked]
