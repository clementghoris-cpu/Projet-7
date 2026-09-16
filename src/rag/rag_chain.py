import logging
from pathlib import Path
from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_mistralai import ChatMistralAI

from src.config.config import app_config, api_keys_config
from src.data.vector_store import VectorStoreManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RAGChainManager:
    def __init__(self):
        """Initialise la chaine RAG avec le vector store, le fichier de prompt et le LLM Mistral"""
        self.vector_store_manager = VectorStoreManager()
        self.prompt_template_str = self._load_prompt_template()
        self.prompt = ChatPromptTemplate.from_template(self.prompt_template_str)
        self.similarity_threshold = app_config.indexer.similarity_threshold

        logging.info(f"Initialisation du LLM Mistral avec le modèle : {app_config.models.llm_model}")
        self.llm = ChatMistralAI(
            model = app_config.models.llm_model,
            api_key = api_keys_config.mistral,
            temperature = 0.2
        )

        self.chain = self._build_chain()

    def _load_prompt_template(self) -> str:
        """Charge le contenu du fichier texte de prompt"""
        path = Path(app_config.paths.rag_prompt_file)

        if not path.exists():
            error_msg = f"Le fichier de prompt est introuvable à l'emplacement : {path.resolve()}"
            logging.error(error_msg)
            raise FileNotFoundError(error_msg)

        try:
            with open(path, 'r', encoding="utf-8") as file:
                template_content = file.read()

            logging.info(f"Prompt chargé avec succès depuis : {path}")
            return template_content
        except Exception as e:
            logging.error(f"Erreur lors de la lecture du fichier de prompt {path} : {e}")
            raise

    def _retrieve_relevant_chunks(self, query: str, top_k: int = None, threshold: float = None) -> List[Dict[str, Any]]:
        """Recherche les chunks les plus similaires dans l'index FAISS."""
        k = top_k or app_config.indexer.search_k
        similarity_threshold = threshold or self.similarity_threshold

        if not self.vector_store_manager.index or self.vector_store_manager.index.ntotal == 0:
            logging.warning("L'index FAISS est vide ou non initialisé.")
            return []

        # Vectorisation de la requête utilisateur
        query_embedding = self.vector_store_manager.embedder.embed_query(query)
        import numpy as np
        query_vector = np.array([query_embedding]).astype("float32")
        import faiss
        faiss.normalize_L2(query_vector)

        # Recherche dans l'index
        distances, indices = self.vector_store_manager.index.search(query_vector, k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1 and idx >= len(self.vector_store_manager.document_chunks):
                continue

            score = float(dist)
            if score >= similarity_threshold:
                chunk = self.vector_store_manager.document_chunks[idx].copy()
                chunk["score"] = score
                results.append(chunk)

        return results

    def _format_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Formate les chunks extraits pour les injecter proprement dans le prompt."""
        if not chunks:
            return "Aucun événement pertinent trouvé."

        formatted_chunks = []
        for i, chunk in enumerate(chunks, 1):
            text = chunk.get("text", "")
            meta = chunk.get("metadata", {})
            url = meta.get("canonicalurl", "N/A")
            formatted_chunks.append(f"--- Événement {i} ---\n{text}\nURL d'information: {url}\n")

        return "\n".join(formatted_chunks)

    def _build_chain(self):
        """Assemble la chaîne LangChain (LCEL) pour le RAG."""
        def get_context(input_data: Dict[str, Any]) -> str:
            query = input_data["question"]
            retrieved_chunks = self._retrieve_relevant_chunks(query)
            return self._format_context(retrieved_chunks)

        chain = (
            {
                "context": get_context,
                "question": lambda x: x["question"]
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        return chain

    def answer_question(self, question: str) -> Dict[str, Any]:
        """Traite une question utilisateur, gère les cas limites et retourne la réponse enrichie avec métadonnées."""
        if not question or not question.strip():
            return {
                "question": question,
                "answer": "Veuillez poser une question valide.",
                "sources": []
            }

        try:
            # Récupération des données sources pour traçabilité
            retrieved_chunks = self._retrieve_relevant_chunks(question)
            
            # Génération de la réponse via LangChain
            response_text = self.chain.invoke({"question": question})

            # Extraction des métadonnées des sources
            if response_text.strip().lower().startswith("désolé"):
                sources = []
            else:
                sources = [chunk.get("metadata", {}) for chunk in retrieved_chunks]

            return {
                "question": question,
                "answer": response_text,
                "sources": sources
            }
        except Exception as e:
            logging.error(f"Erreur lors de la génération de la réponse RAG : {e}")
            return {
                "question": question,
                "answer": "Une erreur technique est survenue lors du traitement de votre demande.",
                "sources": []
            }

if __name__ == "__main__":
    rag_manager = RAGChainManager()
    sample_query = "Quels événements sont prévus durant le mois de septembre?"
    result = rag_manager.answer_question(sample_query)
    
    print("\n=== TEST RAG CHAIN ===")
    print(f"Question: {result['question']}")
    print(f"Réponse:\n{result['answer']}")
    print(f"Nombre de sources utilisées: {len(result['sources'])}")