import re
from dataclasses import dataclass


@dataclass(frozen=True)
class TextChunk:
    text: str
    chunk_index: int


def normalize_text(text: str) -> str:
    """Normaliza espaços preservando quebras de parágrafo."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _tail_by_chars(text: str, max_chars: int) -> str:
    if max_chars <= 0 or not text:
        return ""
    tail = text[-max_chars:]
    first_space = tail.find(" ")
    if first_space > 0:
        tail = tail[first_space + 1 :]
    return tail.strip()


def _split_long_block(block: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Divide um bloco muito grande por janela deslizante de caracteres."""
    step = chunk_size - chunk_overlap
    chunks: list[str] = []
    start = 0

    while start < len(block):
        end = start + chunk_size
        chunk = block[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(block):
            break
        start += step

    return chunks


def split_text(text: str, chunk_size: int = 1200, chunk_overlap: int = 180) -> list[TextChunk]:
    """Divide texto em chunks com sobreposição.

    A estratégia prioriza parágrafos, o que costuma funcionar melhor em RAG do que
    cortar cegamente por quantidade de caracteres. Blocos muito grandes são
    divididos por janela deslizante.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap deve ser menor que chunk_size.")

    text = normalize_text(text)
    if not text:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    raw_chunks: list[str] = []
    current = ""

    def flush_current() -> None:
        nonlocal current
        if current.strip():
            raw_chunks.append(current.strip())
        current = ""

    for paragraph in paragraphs:
        if len(paragraph) > chunk_size:
            flush_current()
            raw_chunks.extend(_split_long_block(paragraph, chunk_size, chunk_overlap))
            continue

        candidate = paragraph if not current else f"{current}\n\n{paragraph}"
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        previous_tail = _tail_by_chars(current, chunk_overlap)
        flush_current()
        if previous_tail:
            current = f"{previous_tail}\n\n{paragraph}"
        else:
            current = paragraph

    flush_current()

    return [TextChunk(text=chunk, chunk_index=index) for index, chunk in enumerate(raw_chunks)]
