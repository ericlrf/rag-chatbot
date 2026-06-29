from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.config import get_settings
from app.core.document_loader import SUPPORTED_EXTENSIONS
from app.core.feedback import FeedbackService
from app.core.openai_client import OpenAIService
from app.core.rag import RAGService
from app.core.vectorstore import VectorStoreService
from app.schemas import (
    ChatRequest,
    ChatResponse,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
)
from app.security import verify_api_key
from app.utils.logging import configure_logging

configure_logging()
settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    description="API de chatbot RAG para consulta a documentos com LLMs.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

openai_service = OpenAIService(settings)
vectorstore_service = VectorStoreService(settings, openai_service)
rag_service = RAGService(settings, openai_service, vectorstore_service)
feedback_service = FeedbackService(settings)

frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    index_path = frontend_dir / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    raise HTTPException(status_code=404, detail="Frontend não encontrado.")


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        app_name=settings.app_name,
        collection_name=settings.collection_name,
        indexed_chunks=vectorstore_service.count(),
    )


@app.post(
    "/api/upload",
    dependencies=[Depends(verify_api_key)],
)
async def upload_document(
    file: UploadFile = File(...),
    ingest_after_upload: bool = False,
) -> dict[str, str | int | bool]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Nome de arquivo ausente.")

    file_name = Path(file.filename).name
    suffix = Path(file_name).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise HTTPException(
            status_code=400,
            detail=f"Formato não suportado. Use: {allowed}.",
        )

    content = await file.read()
    max_size_mb = 20
    if len(content) > max_size_mb * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail=f"Arquivo muito grande. Limite: {max_size_mb} MB.",
        )

    destination = settings.documents_dir / file_name
    destination.write_bytes(content)

    indexed_chunks = 0
    if ingest_after_upload:
        stats = vectorstore_service.ingest_directory(
            directory=settings.documents_dir,
            reset_collection=False,
        )
        indexed_chunks = stats.indexed_chunks

    return {
        "message": "Arquivo enviado com sucesso.",
        "file_name": file_name,
        "saved_path": str(destination),
        "ingest_after_upload": ingest_after_upload,
        "indexed_chunks": indexed_chunks,
    }


@app.post(
    "/api/ingest",
    response_model=IngestResponse,
    dependencies=[Depends(verify_api_key)],
)
def ingest(request: IngestRequest) -> IngestResponse:
    directory = Path(request.directory) if request.directory else settings.documents_dir
    try:
        stats = vectorstore_service.ingest_directory(
            directory=directory,
            reset_collection=request.reset_collection,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return IngestResponse(
        indexed_chunks=stats.indexed_chunks,
        indexed_files=stats.indexed_files,
        collection_name=settings.collection_name,
        message="Indexação concluída." if stats.indexed_chunks else "Nenhum documento encontrado para indexar.",
    )


@app.post(
    "/api/chat",
    response_model=ChatResponse,
    dependencies=[Depends(verify_api_key)],
)
def chat(request: ChatRequest) -> ChatResponse:
    if vectorstore_service.count() == 0:
        raise HTTPException(
            status_code=400,
            detail="Nenhum documento indexado. Execute /api/ingest ou python -m scripts.ingest primeiro.",
        )

    answer, sources, used_top_k = rag_service.answer(
        question=request.question,
        top_k=request.top_k,
        history=request.history,
    )
    return ChatResponse(
        answer=answer,
        sources=sources,
        model=settings.openai_chat_model,
        used_top_k=used_top_k,
    )


@app.post(
    "/api/feedback",
    response_model=FeedbackResponse,
    dependencies=[Depends(verify_api_key)],
)
def feedback(request: FeedbackRequest) -> FeedbackResponse:
    feedback_service.save(request)
    return FeedbackResponse(message="Feedback registrado com sucesso.")
