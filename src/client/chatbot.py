from datetime import date

import requests
import streamlit as st
from requests.exceptions import HTTPError

from src.config.config import api_settings

# Configuration de l'URL de votre API FastAPI (par défaut sur localhost:8000)
API_URL = f"{api_settings.api_url}:{api_settings.api_port}"

# Limite d'échanges (questions + réponses) à conserver dans la mémoire contextuelle
MAX_HISTORY_TURNS = 3 

st.set_page_config(
    page_title="Puls-Events Chatbot",    
    layout="wide"
)

def display_sources(sources: list):
    """Affiche les sources OpenAgenda de manière structurée dans un expander."""
    if not sources:
        return
        
    with st.expander("📚 Sources utilisées"):
        for i, src in enumerate(sources):
            # Extraction des variables
            title = src.get("title", "Événement sans titre")
            city = src.get("city", "Ville non spécifiée")
            url = src.get("canonicalurl")
            raw_dates = src.get("event_dates", [])
            
            # Formate les dates pour un affichage propre (ex: 17/02/2027)
            formatted_dates = []
            for d in raw_dates:
                try:
                    formatted_dates.append(date.fromisoformat(d).strftime("%d/%m/%Y"))
                except ValueError:
                    formatted_dates.append(d)
            
            dates_str = ", ".join(formatted_dates) if formatted_dates else "Date non disponible"

            # Séparateur visuel entre les sources
            if i > 0:
                st.divider()

            # Mise en page sur deux colonnes (Infos + Lien)
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"**📌 {title}**")
                st.caption(f"📍 **Ville :** {city} | 📅 **Dates :** {dates_str}")
                
            with col2:
                if url:
                    st.link_button("🔗 Voir l'événement", url, use_container_width=True)


# ------------------------------------------------------------------------------
# Barre latérale : Métadonnées et statut de l'API
# ------------------------------------------------------------------------------
with st.sidebar:
    st.title("Puls-Events RAG")
    st.markdown("---")
    
    # Vérification du statut de l'API (/health)
    try:
        health_res = requests.get(f"{API_URL}/health", timeout=5)
        if health_res.status_code == 200:
            health_data = health_res.json()
            st.success("🟢 API Connectée")
            st.write(f"**Index FAISS chargé :** {'Oui' if health_data.get('vector_store_loaded') else 'Non'}")
            st.write(f"**Vecteurs en mémoire :** {health_data.get('total_vectors', 0)}")
        else:
            st.error("🔴 API Inaccessible")
    except Exception as e:   # noqa: BLE001
        st.error(f"🔴 Erreur de connexion : {e}")

    st.markdown("---")
    
    # Récupération des métadonnées (/metadata)
    try:
        meta_res = requests.get(f"{API_URL}/metadata", timeout=5)
        meta_res.raise_for_status()

        if meta_res.status_code == 200:
            meta_data = meta_res.json()
            st.subheader("Configuration RAG")
            st.write(f"**Modèle LLM :** {meta_data.get('llm_model')}")
            st.write(f"**Modèle Embeddings :** {meta_data.get('embeddings_model')}")
            st.write(f"**Chunks totaux :** {meta_data.get('total_chunks')}")
    except HTTPError as e:
        print(f"Erreur status HTTP : {e}")

    st.markdown("---")
    # Bouton pour vider l'historique
    if st.button("Effacer la conversation"):
        st.session_state.messages = []
        st.rerun()

# ------------------------------------------------------------------------------
# Interface de Chat principale
# ------------------------------------------------------------------------------
st.title("Assistant Recommandation d'Événements")
st.caption("Posez vos questions sur les événements culturels disponibles !")

# Initialisation de l'historique de session Streamlit
if "messages" not in st.session_state:
    st.session_state.messages = []

# Affichage de l'historique de la discussion
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Si le message contient des sources (réponse de l'assistant)
        if message.get("sources"):
            display_sources(message["sources"])

# Entrée utilisateur
if prompt := st.chat_input("Ex: Quels sont les concerts prévus ce week-end ?"):
    
    # Ajouter et afficher le message de l'utilisateur
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
   
    # Appel à l'API RAG FastAPI
    with st.chat_message("assistant"), st.spinner("Recherche d'événements en cours..."):
        try:
            response = requests.post(
                f"{API_URL}/ask",
                json={"question": prompt},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Récupération de la réponse et des sources retournées par votre API
                answer = data.get("answer", data.get("response", "Pas de réponse reçue."))
                sources = data.get("sources", [])
                
                # Affichage de la réponse
                st.markdown(answer)
                
                if sources:
                    display_sources(sources)

                # Sauvegarde dans la session Streamlit
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

            else:
                error_detail = response.json().get("detail", "Erreur lors de la requête.")
                st.error(f"Erreur API ({response.status_code}) : {error_detail}")

        except requests.exceptions.RequestException as e:
            st.error(f"Impossible de contacter l'API : {e}")