import json
import logging
from bs4 import BeautifulSoup
from langchain_core.documents import Document

# --- 
# Les méthodes permettent de :
#  - lire le fichier des évènement openagenda
#  - récupérer les variables intéressantes de chaque évènement :
#       - canonicalurl
#       - title_fr
#       - longdescription_fr
#       - daterange_fr
#       - timings
#       - location_name
#       - location_address
#       - location_district
#       - location_postalcode
#       - location_city
#       - location_department)
# - les nettoyer
# - les transformer en Document prêts à être convertis en embeddings pour sauvegarde dans l'index Faiss
# ---

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clean_html(html_content : str) -> str:
    """Supprime les balises HTML et les espaces superflus"""
    if not html_content:
        return ""

    soup = BeautifulSoup(html_content, "html.parser")
    return " ".join(soup.get_text(separator=" ").split())

def build_location_str(item : dict) -> str:
    """Reconstitue une adresse lisible à partir des champs de localisation"""
    location_parts = [
        item.get("location_name"),
        item.get("location_address"),
        item.get("location_district"),
        f"{item.get('location_postalcode', '')} {item.get('location_city', '')}".strip(),
        item.get("location_department"),
    ]

    return ", ".join([p for p in location_parts if p])

def parse_json_data(file_path : str) -> list[Document]:
    """Charge les données JSON et retourne une liste de Documents LangChain"""
    with open(file_path, "r", encoding="utf-8") as file:
        events = json.load(file)

    documents = []
    logging.info(f"Lecture du fichier {file_path}. {len(events)} évènements récupérés.")

    for item in events:
        clean_desc = clean_html(item.get("longdescription_fr", ""))
        full_location = build_location_str(item)

        keywords = item.get("keywords_fr") or []
        keywords_str = (
            ", ".join(keywords) if isinstance(keywords, list) else str(keywords)
        )

        page_content = f"""Titre: {item.get('title_fr', '')}
                            Mots-clés: {keywords_str}
                            Description: {clean_desc}
                            Date: {item.get('daterange_fr', '')}
                            Lieu: {full_location}"""

        metadata = {
            "uid": str(item.get("uid", "")),
            "canonicalurl": item.get("canonicalurl", ""),
            "title": item.get("title_fr", ""),
            "city": item.get("location_city", ""),
            "postalcode": item.get("location_postalcode", ""),
            "department": item.get("location_department", ""),
            "daterange": item.get("daterange_fr", ""),
            "firstdate_begin": item.get("firstdate_begin", ""),
        }

        documents.append(Document(page_content = page_content, metadata = metadata))

    return documents