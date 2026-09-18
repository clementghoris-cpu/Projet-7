import argparse
import logging

from src.config.config import app_config
from src.data.vector_store import VectorStoreManager
from src.utils.data_loader import parse_json_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_indexing(events_file_path : str):
    """Exécute le processus d'indexation"""
    logger.info(f"Chargement des évènements depuis le fichier : {events_file_path}")
    documents = parse_json_data(events_file_path)

    if not documents:
        logger.warning("Aucune donnée n'a été chargée. Vérifier le contenu de votre fichier évènements.")
        logger.info("--- Processus d'indexation terminé (aucune donnée traitée) ---")
        return

    logger.info("Initialisation du gestionnaire de Vector Store...")
    vector_store = VectorStoreManager()

    logger.info("Construction de l'index Faiss (cela peut prendre du temps)...")
    vector_store.build_index(documents)

    logger.info("--- Processus d'indexation terminé avec succès ---")
    logger.info(f"Nombre d'évènements traités : {len(documents)}")

    if vector_store.index:
        logger.info(f"Nombre de chunks indexés : {vector_store.index.ntotal}")
    else:
        logger.warning("L'index final n'a pas pu être créé ou est vide.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script d'indexation pour l'application RAG")
    parser.add_argument(
        "--events-file-path",
        type=str,
        default=app_config.paths.openagenda_events,
        help=f"Fichier contenant les évènement venant d'openagenda (par défaut : {app_config.paths.openagenda_events})"
    )

    args = parser.parse_args()
    run_indexing(events_file_path=args.events_file_path)