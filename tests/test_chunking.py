from app.core.chunking import normalize_text, split_text


def test_normalize_text_removes_extra_spaces():
    text = "Olá,   mundo!\n\n\nEste é   um teste."
    assert normalize_text(text) == "Olá, mundo!\n\nEste é um teste."


def test_split_text_returns_chunks_with_indexes():
    text = "\n\n".join([f"Parágrafo {i}: conteúdo de teste." for i in range(30)])
    chunks = split_text(text, chunk_size=180, chunk_overlap=30)
    assert len(chunks) > 1
    assert chunks[0].chunk_index == 0
    assert all(chunk.text for chunk in chunks)


def test_split_text_rejects_invalid_overlap():
    try:
        split_text("abc", chunk_size=100, chunk_overlap=100)
    except ValueError as exc:
        assert "chunk_overlap" in str(exc)
    else:
        raise AssertionError("ValueError esperado")
