from app.config import get_settings
from app.core.openai_client import OpenAIService
from app.core.vectorstore import VectorStoreService


def main() -> None:
    settings = get_settings()
    openai_service = OpenAIService(settings)
    vectorstore = VectorStoreService(settings, openai_service)
    vectorstore.reset_collection()
    print(f"Coleção reiniciada: {settings.collection_name}")


if __name__ == "__main__":
    main()
