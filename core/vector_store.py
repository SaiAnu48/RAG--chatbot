"""Persistent Chroma policy index using cosine-distance retrieval."""
from __future__ import annotations
from pathlib import Path
import re
import chromadb
from models.schemas import PolicyChunk

class PolicyVectorStore:
    """Typed wrapper around a persistent cosine-distance Chroma collection."""
    def __init__(self, directory: str | Path, collection_name: str = "policy_chunks") -> None:
        self._client = chromadb.PersistentClient(path=str(directory))
        self._collection = self._client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
    def index(self, chunks: list[PolicyChunk]) -> None:
        """Idempotently upsert source chunks."""
        self._collection.upsert(ids=[c.id for c in chunks], documents=[c.text for c in chunks], metadatas=[c.chroma_metadata() for c in chunks])
    def search(self, query: str, limit: int = 8, required_tag: str | None = None) -> list[PolicyChunk]:
        """Retrieve policy chunks, with optional tagged subset filtering."""
        result = self._collection.query(query_texts=[query], n_results=50 if required_tag else limit, include=["documents", "metadatas"])
        records = [PolicyChunk(id=i, text=t, document=str(m["document"]), section_or_page=str(m["section_or_page"]), chunk_index=int(m["chunk_index"]), tags=str(m.get("tags", "")).split(",") if m.get("tags") else []) for i, t, m in zip(result["ids"][0], result["documents"][0], result["metadatas"][0])]
        return [record for record in records if required_tag is None or required_tag in record.tags][:limit]

    def lexical_search(self, query: str, limit: int = 8) -> list[PolicyChunk]:
        """Return exact-term matches as a deterministic complement to vector retrieval.

        Policy manuals are generally bounded documents. This fallback protects section
        headers, procedure names, and codes when embedding similarity under-ranks them.
        """
        tokens = {token.lower() for token in re.findall(r"[A-Za-z0-9]{3,}", query)}
        if not tokens:
            return []
        result = self._collection.get(include=["documents", "metadatas"])
        scored: list[tuple[int, PolicyChunk]] = []
        for chunk_id, text, meta in zip(result["ids"], result["documents"], result["metadatas"]):
            score = sum(token in text.lower() for token in tokens)
            if score:
                scored.append((score, PolicyChunk(id=chunk_id, text=text, document=str(meta["document"]), section_or_page=str(meta["section_or_page"]), chunk_index=int(meta["chunk_index"]), tags=str(meta.get("tags", "")).split(",") if meta.get("tags") else [])))
        return [chunk for _, chunk in sorted(scored, key=lambda item: (-item[0], item[1].chunk_index))[:limit]]
