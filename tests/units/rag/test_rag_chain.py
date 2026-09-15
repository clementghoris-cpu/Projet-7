import pytest
import numpy as np
from unittest.mock import MagicMock, patch, mock_open
from langchain_core.runnables import RunnableSequence

from src.rag.rag_chain import RAGChainManager


# ============================================================================
# Fixtures pour isoler les dépendances externes
# ============================================================================

@pytest.fixture
def mock_app_config():
    """Simule la configuration globale de l'application."""
    with patch("src.rag.rag_chain.app_config") as mock_cfg, \
         patch("src.rag.rag_chain.api_keys_config") as mock_keys:
        mock_cfg.models.llm_model = "mistral-small-latest"
        mock_cfg.paths.rag_prompt_file = "/fake/path/prompt.txt"
        mock_cfg.indexer.search_k = 2
        mock_keys.mistral = "fake_api_key"
        yield mock_cfg


@pytest.fixture
def mock_dependencies():
    """Mocke le VectorStoreManager, le prompt file et le LLM Mistral."""
    prompt_content = "Context: {context}\nQuestion: {question}\nAnswer:"

    with patch("src.rag.rag_chain.Path.exists", return_value=True), \
         patch("builtins.open", mock_open(read_data=prompt_content)), \
         patch("src.rag.rag_chain.VectorStoreManager") as mock_vsm_cls, \
         patch("src.rag.rag_chain.ChatMistralAI") as mock_llm_cls:

        # Configuration du VectorStoreManager mocké
        mock_vsm_inst = MagicMock()
        mock_vsm_inst.index.ntotal = 2
        mock_vsm_inst.embedder.embed_query.return_value = [0.1, 0.2, 0.3]
        mock_vsm_inst.document_chunks = [
            {"text": "Événement Concert A", "metadata": {"canonicalurl": "https://example.com/a"}},
            {"text": "Événement Festival B", "metadata": {"canonicalurl": "https://example.com/b"}}
        ]
        # FAISS search retourne des distances et des index : (array([[...]]), array([[0, 1]]))
        mock_vsm_inst.index.search.return_value = (
            np.array([[0.1, 0.2]], dtype="float32"),
            np.array([[0, 1]])
        )
        mock_vsm_cls.return_value = mock_vsm_inst

        # Configuration du LLM mocké
        mock_llm_inst = MagicMock()
        mock_llm_cls.return_value = mock_llm_inst

        yield {
            "vsm": mock_vsm_inst,
            "llm": mock_llm_inst
        }


# ============================================================================
# Tests d'Initialisation & Fichier Prompt
# ============================================================================

def test_init_success():
    """Vérifie que RAGChainManager s'initialise correctement quand le prompt existe."""
    rag_manager = RAGChainManager()
    assert rag_manager.prompt_template_str is not None
    assert rag_manager.chain is not None


@patch("src.rag.rag_chain.Path.exists", return_value=False)
def test_init_missing_prompt_file_raises_error(mock_exists):
    """Vérifie qu'une exception FileNotFoundError est levée si le prompt est absent."""
    with pytest.raises(FileNotFoundError):
        RAGChainManager()


# ============================================================================
# Tests de Récupération & Formatage du Contexte
# ============================================================================

def test_retrieve_relevant_chunks_success(mock_dependencies):
    """Vérifie la récupération et normalisation des vecteurs avec FAISS."""
    rag_manager = RAGChainManager()
    chunks = rag_manager._retrieve_relevant_chunks("Soirée concert", top_k=2)

    assert len(chunks) == 2
    assert chunks[0]["text"] == "Événement Concert A"
    mock_dependencies["vsm"].embedder.embed_query.assert_called_once_with("Soirée concert")
    mock_dependencies["vsm"].index.search.assert_called_once()


def test_retrieve_relevant_chunks_empty_index(mock_dependencies):
    """Vérifie le comportement si l'index FAISS est vide ou non initialisé."""
    mock_dependencies["vsm"].index = None

    rag_manager = RAGChainManager()
    chunks = rag_manager._retrieve_relevant_chunks("Question test")

    assert chunks == []


def test_format_context_with_chunks():
    """Vérifie le formatage textuel des chunks extraits."""
    rag_manager = RAGChainManager()
    chunks = [
        {"text": "Exposition d'art", "metadata": {"canonicalurl": "https://art.com"}},
        {"text": "Match de football", "metadata": {}}  # Sans URL
    ]

    formatted = rag_manager._format_context(chunks)

    assert "--- Événement 1 ---" in formatted
    assert "Exposition d'art" in formatted
    assert "URL d'information: https://art.com" in formatted
    assert "URL d'information: N/A" in formatted


def test_format_context_empty():
    """Vérifie le message retourné si aucun chunk n'est fourni."""
    rag_manager = RAGChainManager()
    formatted = rag_manager._format_context([])
    assert formatted == "Aucun événement pertinent trouvé."


# ============================================================================
# Tests de Génération de Réponse (answer_question)
# ============================================================================

def test_answer_question_valid_input(mock_dependencies):
    """Vérifie le flux complet de réponse à une question valide."""
    rag_manager = RAGChainManager()

    # Mock de l'exécution LCEL de la chaîne
    with patch.object(RunnableSequence, "invoke", return_value="Voici les événements de septembre...") as mock_invoke:
        result = rag_manager.answer_question("Quels événements en septembre ?")

        assert result["question"] == "Quels événements en septembre ?"
        assert result["answer"] == "Voici les événements de septembre..."
        assert len(result["sources"]) == 2
        assert result["sources"][0]["canonicalurl"] == "https://example.com/a"
        mock_invoke.assert_called_once_with({"question": "Quels événements en septembre ?"})


@pytest.mark.parametrize("empty_query", ["", "   ", None])
def test_answer_question_empty_or_invalid_input(empty_query):
    """Vérifie la gestion des questions vides ou invalides."""
    rag_manager = RAGChainManager()
    result = rag_manager.answer_question(empty_query)

    assert result["answer"] == "Veuillez poser une question valide."
    assert result["sources"] == []


def test_answer_question_handles_exception():
    """Vérifie qu'une erreur durant l'exécution de la chaîne est capturée proprement."""
    rag_manager = RAGChainManager()

    with patch.object(RunnableSequence, "invoke", side_effect=Exception("Erreur API Mistral")):
        result = rag_manager.answer_question("Une question ?")

        assert result["answer"] == "Une erreur technique est survenue lors du traitement de votre demande."
        assert result["sources"] == []