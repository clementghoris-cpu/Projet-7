import json
import logging
import os

import requests
from requests.exceptions import HTTPError

from src.config.config import openagenda_settings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

WHERE_FILTER = (
    f"{openagenda_settings.location_filter} "
    f"AND firstdate_begin >= '{openagenda_settings.start_date}' "
    f"AND firstdate_begin <= '{openagenda_settings.end_date}'"
)

PAGE_SIZE = 100

def _fetch_events() -> list[dict]:
    """Requête l'API open agenda pour récupérer les événements filtré région/département/ville et dates"""

    events = []
    offset = 0

    logger.info("Requête API OpenAgenda pour récupérer les événements...")
    logger.info(f"Filtre appliqué : {WHERE_FILTER}")

    while len(events) < openagenda_settings.max_events:
        params = {
            "where": WHERE_FILTER,
            "limit": PAGE_SIZE,
            "offset": offset
        }

        response = requests.get(openagenda_settings.api_url, params=params, timeout=60)
        response.raise_for_status()

        batch = response.json().get("results", [])
        if not batch:
            break

        events.extend(batch)
        offset += PAGE_SIZE
        logger.info(f"{len(events)} événements récupérés")

    logger.info(f"Requête API OpenAgenda terminé, {len(events)} événements récupérés.")
    return events[:openagenda_settings.max_events]

def update_events(events_file_path : str) -> bool:
    """Mise à jour des événement OpenAgenda. Retourne True si la mise à jour a été effectuée"""
    try:
        logger.info("Mise à jour des événements OpenAgenda...")
        events = _fetch_events()

        folder_path = os.path.dirname(events_file_path)
        
        if folder_path:
            os.makedirs(folder_path, exist_ok=True)        

        with open(events_file_path, 'w', encoding='utf-8') as file:
            json.dump(events, file, ensure_ascii=True, indent=4)

        logger.info("Mise à jour des événements OpenAgenda terminée.")
        return True

    except HTTPError:
        logger.exception("Erreur survenue lors de la mise à jour des événements")
        return False


if __name__ == "__main__":    
    print("Récupération des événements...")
    file_path = "ressources/evenements-publics-openagenda.json"
    folder_path = os.path.dirname(file_path)

    if folder_path:
        os.makedirs(folder_path, exist_ok=True)

    update_events(file_path)