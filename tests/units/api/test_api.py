from unittest.mock import MagicMock, patch

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from src.api.api import app

# ============================================================================
# Fixtures & Mocks Setup
# ============================================================================
@pytest.fixture
def client():
    """Client de test FastAPI."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def mock_rag_manager():
    """Fixture créant un mock complet de RAGChainManager."""
    rag_mock = MagicMock()

    # Configuration du VectorStore mocké
    rag_mock.vector_store_manager.index.ntotal = 150
    rag_mock.vector_store_manager.document_chunks = [MagicMock()] * 150

    # Configuration de la réponse par défaut de la méthode answer_question
    rag_mock.answer_question.return_value = {
        "question": "Quels sont les concerts ce week-end ?",
        "answer": "Voici les événements disponibles en septembre...",
        "sources": [
            {"title": "Concert Jazz", "url": "https://example.com/jazz"}
        ],
    }
    return rag_mock


# ============================================================================
# Tests Endpoint : GET /health
# ============================================================================
def test_health_check_success(client, mock_rag_manager):
    """Teste /health quand le RAG et l'index FAISS sont bien initialisés."""
    with patch("src.api.api.rag_manager", mock_rag_manager):
        response = client.get("/health")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "ok"
        assert data["vector_store_loaded"] is True
        assert data["total_vectors"] == 150


def test_health_check_not_loaded(client):
    """Teste /health quand le RAG n'est pas initialisé (None)."""
    with patch("src.api.api.rag_manager", None):
        response = client.get("/health")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "ok"
        assert data["vector_store_loaded"] is False
        assert data["total_vectors"] == 0


# ============================================================================
# Tests Endpoint : GET /metadata
# ============================================================================
def test_get_metadata_success(client, mock_rag_manager):
    """Teste /metadata avec un système RAG valide."""
    with patch("src.api.api.rag_manager", mock_rag_manager):
        response = client.get("/metadata")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["total_chunks"] == 150
        assert "embeddings_model" in data
        assert "llm_model" in data


def test_get_metadata_uninitialized_rag(client):
    """Teste /metadata quand le RAG n'est pas initialisé."""
    with patch("src.api.api.rag_manager", None):
        response = client.get("/metadata")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["total_chunks"] == 0


# ============================================================================
# Tests Endpoint : POST /ask
# ============================================================================
def test_ask_question_success(client, mock_rag_manager):
    """Teste /ask avec une question valide."""
    with patch("src.api.api.rag_manager", mock_rag_manager):
        payload = {"question": "Quels sont les concerts ce week-end ?"}
        response = client.post("/ask", json=payload)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert (
            data["answer"] == "Voici les événements disponibles en septembre..."
        )
        mock_rag_manager.answer_question.assert_called_once_with(
            "Quels sont les concerts ce week-end ?"
        )


def test_ask_question_empty_payload(client, mock_rag_manager):
    """Teste /ask avec une chaîne de caractères vide ou composée d'espaces (HTTP 400)."""
    with patch("src.api.api.rag_manager", mock_rag_manager):
        payload = {"question": "   "}
        response = client.post("/ask", json=payload)

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "La question ne peut pas être vide." in response.json()["detail"]


def test_ask_question_rag_not_initialized(client):
    """Teste /ask quand rag_manager est None (HTTP 503)."""
    with patch("src.api.api.rag_manager", None):
        payload = {"question": "Y a-t-il un festival de théâtre ?"}
        response = client.post("/ask", json=payload)

        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        assert (
            "Le système RAG n'est pas encore initialisé."
            in response.json()["detail"]
        )


def test_ask_question_invalid_body_format(client):
    """Teste /ask avec un body JSON invalide (ex: champ manquant) (HTTP 422)."""
    response = client.post("/ask", json={"query": "Mauvaise clé"})

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT


# ============================================================================
# Tests Endpoint : POST /rebuild
# ============================================================================
@patch("src.api.api.run_indexing")
@patch("src.api.api.RAGChainManager")
def test_rebuild_index_triggers_background_task(
    mock_rag_cls, mock_run_indexing, client
):
    """Teste si /rebuild retourne une réponse 200 immédiate et exécute la tâche de fond."""
    response = client.post("/rebuild")

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["status"] == "processing"
    assert (
        "La reconstruction de l'index FAISS a été démarrée" in data["message"]
    )

    # TestClient exécute automatiquement les BackgroundTasks à la fin de la requête
    mock_run_indexing.assert_called_once()
    mock_rag_cls.assert_called_once()