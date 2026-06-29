import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb

from app.config import Settings
from app.core.chunking import split_text
from app.core.document_loader import iter_document_files, load_document
from app.core.openai_client import OpenAIService


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    metadata: dict[str, Any]
    distance: float | None


@dataclass(frozen=True)
class IngestStats:
    indexed_chunks: int
    indexed_files: int


class VectorStoreService:
    """Serviço de persistência e busca vetorial usando ChromaDB local."""

    def __init__(self, settings: Settings, openai_service: OpenAIService):
        self.settings = settings
        self.openai = openai_service
        self.client = chromadb.PersistentClient(path=str(settings.chroma_path))
        self.collection = self.client.get_or_create_collection(
            name=settings.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def reset_collection(self) -> None:
        try:
            self.client.delete_collection(name=self.settings.collection_name)
        except Exception:
            # Chroma pode lançar exceção se a coleção ainda não existir.
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.settings.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        return int(self.collection.count())

    def ingest_directory(self, directory: Path, reset_collection: bool = False) -> IngestStats:
        if reset_collection:
            self.reset_collection()

        files = iter_document_files(directory)
        all_ids: list[str] = []
        all_texts: list[str] = []
        all_metadatas: list[dict[str, Any]] = []
        indexed_files = 0

        for file_path in files:
            loaded_parts = load_document(file_path)
            file_had_chunks = False

            for part in loaded_parts:
                chunks = split_text(
                    part.text,
                    chunk_size=self.settings.chunk_size,
                    chunk_overlap=self.settings.chunk_overlap,
                )
                for chunk in chunks:
                    chunk_id = _make_chunk_id(
                        file_path=file_path,
                        page=part.page,
                        chunk_index=chunk.chunk_index,
                        text=chunk.text,
                    )
                    metadata = {
                        "file_name": file_path.name,
                        "path": str(file_path),
                        "chunk_index": chunk.chunk_index,
                    }
                    if part.page is not None:
                        metadata["page"] = part.page
                    all_ids.append(chunk_id)
                    all_texts.append(chunk.text)
                    all_metadatas.append(metadata)
                    file_had_chunks = True

            if file_had_chunks:
                indexed_files += 1

        if not all_texts:
            return IngestStats(indexed_chunks=0, indexed_files=0)

        embeddings = self.openai.embed_texts(all_texts)
        self.collection.upsert(
            ids=all_ids,
            embeddings=embeddings,
            documents=all_texts,
            metadatas=all_metadatas,
        )
        return IngestStats(indexed_chunks=len(all_texts), indexed_files=indexed_files)

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        query_embedding = self.openai.embed_texts([query])[0]
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        retrieved: list[RetrievedChunk] = []
        for doc, metadata, distance in zip(documents, metadatas, distances, strict=False):
            if not doc:
                continue
            retrieved.append(
                RetrievedChunk(
                    text=doc,
                    metadata=metadata or {},
                    distance=float(distance) if distance is not None else None,
                )
            )
        return retrieved


def _make_chunk_id(file_path: Path, page: int | None, chunk_index: int, text: str) -> str:
    raw = f"{file_path.resolve()}::{page}::{chunk_index}::{text[:200]}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
