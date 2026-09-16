import json
import logging
from bs4 import BeautifulSoup
from datetime import datetime
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

def format_iso_to_french(iso_str: str) -> str:
    """Convertit '2026-02-09T09:30:00+01:00' en 'Lundi 9 février 2026 à 09h30'"""
    if not iso_str:
        return ""
    try:
        dt = datetime.fromisoformat(iso_str)
        months = ["janvier", "février", "mars", "avril", "mai", "juin", 
                  "juillet", "août", "septembre", "octobre", "novembre", "décembre"]
        days = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]
        
        day_name = days[dt.weekday()]
        month_name = months[dt.month - 1]
        
        return f"{day_name} {dt.day} {month_name} {dt.year} à {dt.strftime('%Hh%M')}"
    except ValueError:
        return iso_str

def parse_timings(timings_raw) -> tuple[list[str], list[str]]:
    """Extrait tous les créneaux sous forme lisible et sous forme de dates YYYY-MM-DD pour le filtrage."""
    if not timings_raw:
        return [], []
    
    if isinstance(timings_raw, str):
        try:
            timings_raw = json.loads(timings_raw)
        except json.JSONDecodeError:
            return [], []
            
    readable_slots = []
    iso_dates = []
    
    for slot in timings_raw:
        begin = slot.get("begin")
        end = slot.get("end")
        if begin:
            readable_start = format_iso_to_french(begin)
            end_time = datetime.fromisoformat(end).strftime("%Hh%M") if end else ""
            readable_slots.append(f"{readable_start} - {end_time}" if end_time else readable_start)
            iso_dates.append(begin[:10]) # Format YYYY-MM-DD
            
    return readable_slots, list(set(iso_dates))

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

        # Traitement des dates
        readable_slots, dates_list = parse_timings(item.get("timings"))
        
        start_date_readable = format_iso_to_french(item.get("firstdate_begin"))
        end_date_readable = format_iso_to_french(item.get("lastdate_end"))
        
        # Construction d'un bloc de dates complet pour le LLM
        if len(readable_slots) > 1:
            dates_text = f"Événement sur plusieurs créneaux du {start_date_readable} au {end_date_readable}.\nCréneaux exacts:\n- " + "\n- ".join(readable_slots)
        else:
            dates_text = f"Le {start_date_readable}"

        page_content = f"""Titre: {item.get('title_fr', '')}
                            Mots-clés: {keywords_str}
                            Description: {clean_desc}
                            Dates et Horaires: {dates_text}
                            Lieu: {full_location}"""

        metadata = {
            "uid": str(item.get("uid", "")),
            "canonicalurl": item.get("canonicalurl", ""),
            "title": item.get("title_fr", ""),
            "city": item.get("location_city", ""),
            "postalcode": item.get("location_postalcode", ""),
            "department": item.get("location_department", ""),
            "firstdate_begin": item.get("firstdate_begin", ""),
            "lastdate_end": item.get("lastdate_end", ""),
            "event_dates": dates_list,  # Liste des dates YYYY-MM-DD couverte par l'événement
            "is_multiday": len(readable_slots) > 1
        }

        documents.append(Document(page_content = page_content, metadata = metadata))

    return documents