from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pytest
from langchain_core.documents import Document

from src.data.vector_store import VectorStoreManager


@pytest.fixture
def mock_app_config():
    """Simule la configuration globale de l'application."""
    with patch("src.data.vector_store.app_config") as mock_cfg:
        mock_cfg.models.embeddings_model = "sentence-transformers/all-MiniLM-L6-v2"
        mock_cfg.paths.faiss_index_file = "/fake/path/faiss.index"
        mock_cfg.paths.document_chunck_file = "/fake/path/chunks.pkl"
        mock_cfg.indexer.chunk_size = 100
        mock_cfg.indexer.chunk_overlap = 10
        mock_cfg.indexer.embedding_batch_size = 2
        yield mock_cfg


@pytest.fixture
def mock_dependencies(mock_app_config):
    """Mocke les dépendances lourdes : torch, HuggingFaceEmbeddings et l'I/O fichier au démarrage."""
    with patch("src.data.vector_store.torch.cuda.is_available", return_value=False), \
         patch("src.data.vector_store.HuggingFaceEmbeddings") as mock_embedder_cls, \
         patch("os.path.exists", return_value=False):
        
        mock_embedder_inst = MagicMock()
        # Simulation d'un embedding de dimension 4
        mock_embedder_inst.embed_documents.side_effect = lambda texts: [[0.1, 0.2, 0.3, 0.4] for _ in texts]
        mock_embedder_inst.embed_query.return_value = [0.1, 0.2, 0.3, 0.4]
        mock_embedder_cls.return_value = mock_embedder_inst
        
        yield mock_embedder_inst


# ============================================================================
# Tests d'Initialisation et Chargement
# ============================================================================

def test_init_without_existing_files(mock_dependencies):
    """Vérifie que l'initialisation crée une instance vide si aucun fichier n'existe."""
    manager = VectorStoreManager()
    assert manager.index is None
    assert manager.document_chunks == []


@patch("src.data.vector_store.faiss.read_index")
@patch("builtins.open", new_callable=mock_open)
@patch("src.data.vector_store.pickle.load")
@patch("os.path.exists", return_value=True)
def test_init_with_existing_files(mock_exists, mock_pickle, mock_file, mock_read_index, mock_dependencies):
    """Vérifie le chargement réussi de l'index et des chunks existants."""
    dummy_index = MagicMock()
    dummy_index.ntotal = 5
    mock_read_index.return_value = dummy_index
    mock_pickle.return_value = [{"id": "chunk_0", "text": "hello"}]

    manager = VectorStoreManager()

    assert manager.index == dummy_index
    assert len(manager.document_chunks) == 1
    mock_read_index.assert_called_once()


# ============================================================================
# Tests de Découpage de Documents (_split_documents_to_chunks)
# ============================================================================

def test_split_documents_to_chunks(mock_dependencies):
    """Vérifie le découpage correct des documents en dictionnaires de chunks."""
    manager = VectorStoreManager()
    docs = [Document(page_content="Ceci est un test de texte long pour vérifier le découpage.", metadata={"source": "test.txt"})]

    chunks = manager._split_documents_to_chunks(docs)

    assert isinstance(chunks, list)
    assert len(chunks) > 0
    assert chunks[0]["id"] == "chunk_0"
    assert "text" in chunks[0]
    assert chunks[0]["metadata"]["source"] == "test.txt"
    assert "start_index" in chunks[0]["metadata"]


# ============================================================================
# Tests de Génération d'Embeddings (_generate_embeddings)
# ============================================================================

def test_generate_embeddings_success(mock_dependencies):
    """Vérifie la génération standard d'embeddings en lots."""
    manager = VectorStoreManager()
    chunks = [
        {"id": "chunk_0", "text": "Texte 1"},
        {"id": "chunk_1", "text": "Texte 2"},
        {"id": "chunk_2", "text": "Texte 3"}
    ]

    embeddings = manager._generate_embeddings(chunks)

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (3, 4)
    assert embeddings.dtype == np.float32


def test_generate_embeddings_empty_chunks(mock_dependencies):
    """Vérifie le comportement lorsqu'aucun chunk n'est fourni."""
    manager = VectorStoreManager()
    assert manager._generate_embeddings([]) is None


def test_generate_embeddings_fallback_on_batch_error(mock_dependencies):
    """Vérifie qu'en cas d'erreur sur un lot, des vecteurs nuls sont ajoutés en fallback."""
    embedder_mock = mock_dependencies
    # Le premier lot échoue, le deuxième réussit
    embedder_mock.embed_documents.side_effect = [Exception("Erreur GPU"), [[0.1, 0.2, 0.3, 0.4]]]

    manager = VectorStoreManager()
    chunks = [
        {"id": "chunk_0", "text": "Texte 1"},  # Lot 1 (batch_size=2)
        {"id": "chunk_1", "text": "Texte 2"},
        {"id": "chunk_2", "text": "Texte 3"}   # Lot 2
    ]

    embeddings = manager._generate_embeddings(chunks)

    assert embeddings is not None
    assert embeddings.shape == (3, 4)
    # Les deux premiers éléments doivent être des vecteurs nuls
    assert np.all(embeddings[0] == 0)
    assert np.all(embeddings[1] == 0)


# ============================================================================
# Tests de Sauvegarde (_save_index_and_chunks)
# ============================================================================

@patch("os.makedirs")
@patch("src.data.vector_store.faiss.write_index")
@patch("builtins.open", new_callable=mock_open)
@patch("src.data.vector_store.pickle.dump")
def test_save_index_and_chunks_success(mock_pickle, mock_file, mock_write_index, mock_makedirs, mock_dependencies):
    """Vérifie que la sauvegarde écrit correctement l'index et les chunks."""
    manager = VectorStoreManager()
    manager.index = MagicMock()
    manager.document_chunks = [{"id": "chunk_0", "text": "test"}]

    manager._save_index_and_chunks()

    mock_makedirs.assert_called()
    mock_write_index.assert_called_once_with(manager.index, "/fake/path/faiss.index")
    mock_pickle.assert_called_once_with(manager.document_chunks, mock_file())


# ============================================================================
# Tests d'Intégration / Construction de l'Index (build_index)
# ============================================================================

@patch.object(VectorStoreManager, "_save_index_and_chunks")
def test_build_index_success(mock_save, mock_dependencies):
    """Vérifie la construction complète d'un index FAISS à partir de documents."""
    manager = VectorStoreManager()
    docs = [Document(page_content="Document de test.", metadata={"source": "doc1.txt"})]

    manager.build_index(docs)

    assert manager.index is not None
    assert manager.index.ntotal > 0
    assert len(manager.document_chunks) > 0
    mock_save.assert_called_once()


@patch("os.remove")
@patch("os.path.exists", return_value=True)
def test_build_index_failure_cleans_up(mock_exists, mock_remove, mock_dependencies):
    """Vérifie qu'en cas d'échec des embeddings, l'état est réinitialisé et les fichiers nettoyés."""
    manager = VectorStoreManager()
    
    # Simuler un échec de la génération d'embeddings
    with patch.object(manager, "_generate_embeddings", return_value=None):
        docs = [Document(page_content="Document de test.")]
        manager.build_index(docs)

        assert manager.index is None
        assert manager.document_chunks == []
        assert mock_remove.call_count == 2  # Supprime le fichier index et le fichier chunks