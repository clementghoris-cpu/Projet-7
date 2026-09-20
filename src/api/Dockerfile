FROM python:3.12-slim

# Variables d'environnement pour Python :
# - PYTHONUNBUFFERED=1 : Affiche directement les logs sans bufferisation (indispensable sous Docker)
# - PYTHONDONTWRITEBYTECODE=1 : Évite de créer des fichiers .pyc inutiles
# - PYTHONPATH=/app : Garantit que Python trouve les dossiers 'src' et 'config'
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app
    
WORKDIR /app

ENV PATH="/app/.venv/bin:$PATH"

# Installation du package manager uv directement depuis le binaire officiel
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-cache --no-dev --extra api --extra rag --no-install-project

COPY src/ ./src/

RUN uv sync --frozen --no-cache --no-dev --extra api --extra rag

CMD ["uvicorn", "src.api.api:app", "--host", "0.0.0.0", "--port", "8000"]
