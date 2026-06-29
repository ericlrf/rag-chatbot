from fastapi import Header, HTTPException, status

from app.config import get_settings


async def verify_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Proteção simples opcional por API key.

    Em desenvolvimento, deixe APP_API_KEY vazio. Em produção, defina APP_API_KEY
    no .env e envie o cabeçalho X-API-Key em cada requisição.
    """
    settings = get_settings()
    if not settings.is_api_key_auth_enabled:
        return

    if not x_api_key or x_api_key != settings.app_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key inválida ou ausente.",
        )
