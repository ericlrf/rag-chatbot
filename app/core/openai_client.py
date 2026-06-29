from collections.abc import Iterable

from openai import OpenAI

from app.config import Settings


class OpenAIService:
    """Wrapper simples para embeddings e geração via SDK oficial da OpenAI."""

    def __init__(self, settings: Settings):
        if not settings.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY não configurada. Copie .env.example para .env e preencha a chave."
            )
        self.settings = settings
        self.client = OpenAI(api_key=settings.openai_api_key)

    def embed_texts(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for batch in _batched(texts, batch_size):
            response = self.client.embeddings.create(
                model=self.settings.openai_embedding_model,
                input=batch,
            )
            embeddings.extend(item.embedding for item in response.data)
        return embeddings

    def generate_answer(self, *, instructions: str, user_input: str) -> str:
        response = self.client.responses.create(
            model=self.settings.openai_chat_model,
            instructions=instructions,
            input=user_input,
            temperature=self.settings.temperature,
            max_output_tokens=self.settings.max_output_tokens,
        )
        return response.output_text.strip()


def _batched(items: list[str], batch_size: int) -> Iterable[list[str]]:
    for index in range(0, len(items), batch_size):
        yield items[index : index + batch_size]
