import os
import requests
import streamlit as st
from src.config.config import api_settings

# Configuration de l'URL de votre API FastAPI (par défaut sur localhost:8000)
API_URL = f"{api_settings.api_url}:{api_settings.api_port}"

# Limite d'échanges (questions + réponses) à conserver dans la mémoire contextuelle
MAX_HISTORY_TURNS = 3 

st.set_page_config(
    page_title="Puls-Events Chatbot",    
    layout="wide"
)

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
    except Exception as e:
        st.error(f"🔴 Erreur de connexion : {e}")

    st.markdown("---")
    
    # Récupération des métadonnées (/metadata)
    try:
        meta_res = requests.get(f"{API_URL}/metadata", timeout=5)
        if meta_res.status_code == 200:
            meta_data = meta_res.json()
            st.subheader("Configuration RAG")
            st.write(f"**Modèle LLM :** {meta_data.get('llm_model')}")
            st.write(f"**Modèle Embeddings :** {meta_data.get('embeddings_model')}")
            st.write(f"**Chunks totaux :** {meta_data.get('total_chunks')}")
    except Exception:
        pass

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
        if "sources" in message and message["sources"]:
            with st.expander("📚 Sources utilisées"):
                for src in message["sources"]:
                    st.write(f"- **{src.get('title', 'Événement')}** ({src.get('location', 'Lieu N/A')})")

# Entrée utilisateur
if prompt := st.chat_input("Ex: Quels sont les concerts prévus ce week-end ?"):
    
    # 1. Ajouter et afficher le message de l'utilisateur
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Préparer l'historique récent (limité aux MAX_HISTORY_TURNS derniers échanges)
    recent_messages = st.session_state.messages[-(MAX_HISTORY_TURNS * 2):]
    
    # Construction du prompt intégrant le contexte / l'historique de la conversation
    # Note : Ajustez le format d'envoi selon ce que votre endpoint /ask attend dans `QueryRequest`
    full_query = prompt
    if len(recent_messages) > 1:
        history_str = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in recent_messages[:-1]])
        full_query = f"Historique de la conversation :\n{history_str}\n\nNouvelle question utilisateur: {prompt}"

    # 3. Appel à l'API RAG FastAPI
    with st.chat_message("assistant"):
        with st.spinner("Recherche d'événements en cours..."):
            try:
                response = requests.post(
                    f"{API_URL}/ask",
                    json={"question": full_query},
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
                        with st.expander("📚 Sources utilisées"):
                            for src in sources:
                                st.write(f"- **{src.get('title', 'Événement')}** ({src.get('location', 'Lieu N/A')})")

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