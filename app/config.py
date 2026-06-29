from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configurações carregadas do ambiente ou de um arquivo .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "V-LAB RAG Chatbot"
    app_env: Literal["development", "test", "production"] = "development"
    app_api_key: str = ""

    openai_api_key: str = Field(default="", min_length=0)
    openai_chat_model: str = "gpt-4.1-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    documents_dir: Path = Path("data/documents")
    chroma_path: Path = Path("storage/chroma")
    collection_name: str = "vlab_documents"
    feedback_path: Path = Path("storage/feedback.jsonl")

    chunk_size: int = Field(default=1200, ge=200, le=8000)
    chunk_overlap: int = Field(default=180, ge=0, le=2000)
    default_top_k: int = Field(default=5, ge=1, le=20)
    max_context_chars: int = Field(default=9000, ge=1000, le=50000)

    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=900, ge=100, le=4000)

    @field_validator("chunk_overlap")
    @classmethod
    def overlap_must_be_smaller_than_chunk(cls, value: int, info):
        chunk_size = info.data.get("chunk_size", 1200)
        if value >= chunk_size:
            raise ValueError("CHUNK_OVERLAP deve ser menor que CHUNK_SIZE.")
        return value

    @property
    def is_api_key_auth_enabled(self) -> bool:
        return bool(self.app_api_key.strip())

    def ensure_directories(self) -> None:
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_path.mkdir(parents=True, exist_ok=True)
        self.feedback_path.parent.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings
