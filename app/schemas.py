from typing import Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1)


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=3, examples=["Quais são os requisitos da vaga?"])
    top_k: int | None = Field(default=None, ge=1, le=20)
    session_id: str | None = Field(default=None)
    history: list[Message] = Field(default_factory=list)


class SourceChunk(BaseModel):
    source_id: int
    file_name: str
    path: str
    page: int | None = None
    chunk_index: int
    score: float | None = None
    excerpt: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    model: str
    used_top_k: int


class IngestRequest(BaseModel):
    directory: str | None = Field(
        default=None,
        description="Diretório com documentos. Se vazio, usa DOCUMENTS_DIR do .env.",
    )
    reset_collection: bool = Field(
        default=False,
        description="Se true, apaga a coleção existente antes de reindexar.",
    )


class IngestResponse(BaseModel):
    indexed_chunks: int
    indexed_files: int
    collection_name: str
    message: str


class HealthResponse(BaseModel):
    status: Literal["ok"]
    app_name: str
    collection_name: str
    indexed_chunks: int


class FeedbackRequest(BaseModel):
    session_id: str | None = None
    question: str
    answer: str
    rating: int = Field(..., ge=1, le=5)
    comment: str | None = None


class FeedbackResponse(BaseModel):
    message: str
