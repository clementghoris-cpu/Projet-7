import logging
import argparse
from src.config.config import app_config
from src.utils.data_loader import parse_json_data
from src.data.vector_store import VectorStoreManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_indexing(events_file_path : str):
    """Exécute le processus d'indexation"""
    logging.info(f"Chargement des évènements depuis le fichier : {events_file_path}")
    documents = parse_json_data(events_file_path)

    if not documents:
        logging.warning("Aucune donnée n'a été chargée. Vérifier le contenu de votre fichier évènements.")
        logging.info("--- Processus d'indexation terminé (aucune donnée traitée) ---")
        return

    logging.info("Initialisation du gestionnaire de Vector Store...")
    vector_store = VectorStoreManager()

    logging.info("Construction de l'index Faiss (cela peut prendre du temps)...")
    vector_store.build_index(documents)

    logging.info("--- Processus d'indexation terminé avec succès ---")
    logging.info(f"Nombre d'évènements traités : {len(documents)}")

    if vector_store.index:
        logging.info(f"Nombre de chunks indexés : {vector_store.index.ntotal}")
    else:
        logging.warning("L'index final n'a pas pu être créé ou est vide.")



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