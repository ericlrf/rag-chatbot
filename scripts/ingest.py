import argparse
from pathlib import Path

from app.config import get_settings
from app.core.openai_client import OpenAIService
from app.core.vectorstore import VectorStoreService
from app.utils.logging import configure_logging


def main() -> None:
    configure_logging()
    settings = get_settings()

    parser = argparse.ArgumentParser(description="Indexa documentos no banco vetorial ChromaDB.")
    parser.add_argument(
        "--path",
        type=str,
        default=str(settings.documents_dir),
        help="Diretório contendo arquivos .pdf, .txt, .md ou .docx.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Apaga a coleção atual antes de reindexar.",
    )
    args = parser.parse_args()

    openai_service = OpenAIService(settings)
    vectorstore = VectorStoreService(settings, openai_service)
    stats = vectorstore.ingest_directory(Path(args.path), reset_collection=args.reset)

    print("Indexação concluída")
    print(f"Arquivos indexados: {stats.indexed_files}")
    print(f"Chunks indexados: {stats.indexed_chunks}")
    print(f"Coleção: {settings.collection_name}")


if __name__ == "__main__":
    main()
