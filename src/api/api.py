import logging
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, HTTPException, status

from src.api.schemas import (
    HealthResponse,
    MetadataResponse,
    QueryRequest,
    QueryResponse,
    RebuildResponse,
)
from src.config.config import app_config
from src.data.indexer import run_indexing
from src.rag.rag_chain import RAGChainManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

rag_manager: RAGChainManager | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gère le cycle de vie de l'application (Démarrage / Extinction).
    """
    global rag_manager
    logger.info("Démarrage de l'API REST et chargement du système RAG...")
    try:
        rag_manager = RAGChainManager()
        logger.info("Système RAG prêt à recevoir des requêtes.")
    except Exception as e:  #noqa: BLE001
        logger.error(f"Erreur lors de l'initialisation du système RAG au démarrage : {e}")

    yield

    print("Extinction de l'API : Nettoyage des ressources...")


app = FastAPI(
    title="Puls-Events RAG API",
    description="API REST exposant le système de RAG (Retrieval-Augmented Generation) pour les recommandations d'événements culturels.",
    version="1.0.0",
    lifespan=lifespan
)

@app.get(
        "/health", 
        response_model=HealthResponse, 
        tags=["System"],
        summary="Vérifie l'état de l'API et du vector store FAISS",
        description="Retourne status 'OK', si le vector store est correctement chargé et le nombre de vecteurs chargés",
        responses= {
            200: {"description": "Statut API"},
        })
def health_check():
    """Vérifie l'état de santé de l'API et du Vector Store FAISS."""
    is_loaded = False
    vector_count = 0
    
    if rag_manager and rag_manager.vector_store_manager and rag_manager.vector_store_manager.index:
        is_loaded = True
        vector_count = rag_manager.vector_store_manager.index.ntotal

    return HealthResponse(
        status="ok",
        vector_store_loaded=is_loaded,
        total_vectors=vector_count
    )

@app.get(
        "/metadata", 
        response_model=MetadataResponse, 
        tags=["System"],
        summary="Afficher les métadonnées du système d'indexation",
        description=("Retourne les informations liées au système d'indexation:\n"
                     "le nombre total de chunks, les modèles llm et embeddings utilisés, "
                     "la taille des chunks ainsi que l'overlap"),
        responses= {
            200: {"description": "métadonnées du système d'indexation"}
        })
def get_metadata():
    """Retourne les métadonnées et statistiques du système d'indexation."""
    total_chunks = 0
    if rag_manager and rag_manager.vector_store_manager:
        total_chunks = len(rag_manager.vector_store_manager.document_chunks)

    return MetadataResponse(
        total_chunks=total_chunks,
        embeddings_model=app_config.models.embeddings_model,
        llm_model=app_config.models.llm_model,
        chunk_size=app_config.indexer.chunk_size,
        chunk_overlap=app_config.indexer.chunk_overlap
    )

@app.post("/ask", 
          response_model=QueryResponse, 
          tags=["RAG"],
          summary="Question utilisateur posée au système RAG",
          description=(
              "1. Envoie une question au pipeline RAG.\n"
              "2. Le pipeline recherche les événements dans l'index FAISS\n"
              "3. Le pipeline retourne une réponse augmentée via le LLM de Mistral"),
          responses={
              200: {"description": "Réponse du Pipeline RAG"},
              503: {"description": "Système RAG non initialisé"},
              400: {"description": "Erreur question vide"},
          })
def ask_question(request: QueryRequest):
    """Reçoit une question utilisateur et retourne la réponse générée par Mistral augmentée des données FAISS."""
    if not rag_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail="Le système RAG n'est pas encore initialisé."
        )
    
    if not request.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="La question ne peut pas être vide."
        )

    response = rag_manager.answer_question(request.question)
    return QueryResponse(**response)

def _rebuild_task():
    """Tâche en arrière-plan pour reconstruire l'index FAISS et réinitialiser le RAG."""
    global rag_manager
    logger.info("Lancement de la reconstruction de l'index FAISS...")
    run_indexing(events_file_path=app_config.paths.openagenda_events, fetch_latest_events=True)
    # Recharge la chaîne RAG avec le nouvel index
    rag_manager = RAGChainManager()
    logger.info("Reconstruction de l'index terminée et RAG rechargé avec succès.")

@app.post(
        "/rebuild", 
        response_model=RebuildResponse, 
        tags=["Admin"],
        summary="Reconstruction de l'index FAISS",
        description=("Requête l'API Openagenda pour récupérer les événement filtré, "
                     "nettoie les données des événements, "
                     "génère les chunks et embeddings, reconstruit l'index FAISS"),
        responses={
            200: {"description": "Message reconstruction de l'index FAISS démarré en arrière plan"}
        })
def rebuild_index(background_tasks: BackgroundTasks):
    """Déclenche la reconstruction de l'index vectoriel FAISS en arrière-plan à partir des données brutes."""
    background_tasks.add_task(_rebuild_task)
    return RebuildResponse(
        status="processing",
        message="La reconstruction de l'index FAISS a été démarrée en arrière-plan."
    )