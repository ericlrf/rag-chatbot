.PHONY: install run ingest reset test docker-up docker-down

install:
	python -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

run:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

ingest:
	python -m scripts.ingest --path data/documents

reset:
	python -m scripts.clear_index

test:
	pytest -q

docker-up:
	docker compose up --build

docker-down:
	docker compose down
