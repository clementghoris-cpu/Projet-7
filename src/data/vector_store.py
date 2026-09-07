import os
import numpy as np
import logging
import pickle
import faiss
from typing import List, Dict, Any
from src.config.config import app_config
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class VectorStoreManager:
    """Gère la création, le chargement et la recherche dans un index Faiss"""

    def __init__(self):
        self.index : faiss.Index | None = None
        self.document_chunks : List[Dict[str, Any]] = []
        self._load_index_and_chunks()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Initialisation du modèle embeddings : {app_config.models.embeddings_model} sur le device '{device}'...")


        model_kwargs = {"device": device, "trust_remote_code": True}
        encode_kwargs = {"normalize_embeddings": True}

        self.embedder = HuggingFaceEmbeddings(
            model_name = app_config.models.embeddings_model,
            model_kwargs = model_kwargs,
            encode_kwargs = encode_kwargs
        )

    def _load_index_and_chunks(self):
        """Charge l'index Faiss et les chunks si les fichiers existent"""
        if os.path.exists(app_config.paths.faiss_index_file) and os.path.exists(app_config.paths.document_chunck_file):
            try:
                logging.info(f"Chargement de l'index Faiss depuis {app_config.paths.faiss_index_file}...")
                self.index = faiss.read_index(app_config.paths.faiss_index_file)

                logging.info(f"Chargement des chunks depuis {app_config.paths.document_chunck_file}...")
                with open(app_config.paths.document_chunck_file, "rb") as file:
                    self.document_chunks = pickle.load(file)

                logging.info(f"Index ({self.index.ntotal} vecteurs) et {len(self.document_chunks)} chunks chargés.")
            except Exception as e:  # noqa: BLE001
                logging.error(f"Erreur lors du chargement de l'index / chunks : {e}")
                self.index = None
                self.document_chunks = []
        else:
            logging.warning("Fichiers d'index Faiss ou de chunks non trouvés. L'index est vide.")

    def _split_documents_to_chunks(self, documents : List[Document]) -> List[Dict[str, Any]]:
        """Découpe les documents en chunks avec métadonnées."""
        logging.info(f"Découpage de {len(documents)} documents en chunks (taille={app_config.indexer.chunk_size}, chevauchement={app_config.indexer.chunk_overlap})...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=app_config.indexer.chunk_size,
            chunk_overlap=app_config.indexer.chunk_overlap,
            length_function=len,
            add_start_index=True,
        )

        all_chunks = []
        split_docs = text_splitter.split_documents(documents)

        for i, chunk in enumerate(split_docs):
            chunk_dict = {                
                "id": f"chunk_{i}",    
                "text": chunk.page_content,
                "metadata": {
                    **chunk.metadata,  
                    "start_index": chunk.metadata.get(
                        "start_index", -1
                    ),  
                },
            }
            all_chunks.append(chunk_dict)

        logging.info(f"Total de {len(all_chunks)} chunks créés.")
        return all_chunks

    def _generate_embeddings(self, chunks : List[Dict[str, Any]]) -> np.ndarray | None:
        """Génère les embeddings pour une liste de chunks via le modèle d'embeddings"""
        if not chunks:
            logging.warning("Aucun chunk fourni pour générer les embeddings.")
            return None

        logging.info(f"Génération des embeddings pour {len(chunks)} chunks (modèle: {app_config.models.embeddings_model})...")
        all_embeddings = []
        total_batches = (len(chunks) + app_config.indexer.embedding_batch_size - 1) // app_config.indexer.embedding_batch_size

        for i in range(0, len(chunks), app_config.indexer.embedding_batch_size):
            batch_num = (i // app_config.indexer.embedding_batch_size) + 1
            batch_chunks = chunks[i:i + app_config.indexer.embedding_batch_size]
            texts_to_embed = [chunk["text"] for chunk in batch_chunks]

            logging.info(f"  Traitement du lot {batch_num}/{total_batches} ({len(texts_to_embed)} chunks)")

            try:
                batch_embeddings = self.embedder.embed_documents(texts_to_embed)
                all_embeddings.extend(batch_embeddings)
            except Exception as e: # noqa: BLE001
                logging.error(f"Erreur lors de la génération d'embeddings (lot {batch_num}): {e}")

                # Gestion de l'erreur : ajout de vecteurs nuls pour ne pas décaler l'index
                num_failed = len(texts_to_embed)
                if all_embeddings:                    
                    dim = len(all_embeddings[0])        # On récupère la dimension à partir du premier vecteur déjà généré
                else:
                    # Si le premier lot échoue, on tente d'obtenir la dimension avec un texte de test
                    try:
                        dim = len(self.embedder.embed_query("test_dim"))
                    except Exception:
                        logging.error("Impossible de déterminer la dimension des embeddings, saut du lot.")
                        continue

                logging.warning(f"Ajout de {num_failed} vecteurs nuls de dimension {dim} pour le lot échoué.")
                all_embeddings.extend([np.zeros(dim, dtype="float32").tolist()] * num_failed)

        if not all_embeddings:
            logging.error("Aucun embedding n'a pu être généré.")
            return None

        # Conversion finale en tableau numpy float32 pour FAISS
        embeddings_array = np.array(all_embeddings).astype("float32")
        logging.info(f"Embeddings générés avec succès. Shape: {embeddings_array.shape}")

        return embeddings_array

    def _save_index_and_chunks(self):
        """Sauvegarde l'index Faiss et la liste des chunks"""
        if self.index is None or not self.document_chunks:
            logging.warning("Tentative de sauvegarde d'un index ou de chunks vides.")
            return

        try:
            # Créer les dossiers parents s'ils n'existent pas
            os.makedirs(os.path.dirname(app_config.paths.faiss_index_file), exist_ok=True)
            os.makedirs(os.path.dirname(app_config.paths.document_chunck_file), exist_ok=True)

            logging.info(f"Sauvegarde de l'index Faiss dans {app_config.paths.faiss_index_file}...")
            faiss.write_index(self.index, app_config.paths.faiss_index_file)

            logging.info(f"Sauvegarde des chunks dans {app_config.paths.document_chunck_file}...")
            with open(app_config.paths.document_chunck_file, 'wb') as file:
                pickle.dump(self.document_chunks, file)

            logging.info("Index et chunks sauvegardés avec succès.")
        except Exception as e: # noqa: BLE001
            logging.error(f"Erreur lors de la sauvegarde de l'index/chunks : {e}")

    def build_index(self, documents : List[Document]):
        """Construit l'index Faiss à partir des documents"""
        if not documents:
            logging.warning("Aucun document fourni pour construire l'index.")
            return

        self.document_chunks = self._split_documents_to_chunks(documents)

        if not self.document_chunks:
            logging.error("Le découpage n'a produit aucun chunk. Impossible de construire l'index.")
            return

        embeddings = self._generate_embeddings(self.document_chunks)

        if embeddings is None or embeddings.shape[0] != len(self.document_chunks):
            logging.error("Problème de génération d'embeddings. Le nombre d'embeddings ne correspond pas au nombre de chunks.")

            # Nettoyage pour éviter un état incohérent
            self.document_chunks = []
            self.index = None

            # Suppression des fichiers potentiellement corrompus
            if os.path.exists(app_config.paths.faiss_index_file):
                os.remove(app_config.paths.faiss_index_file)

            if os.path.exists(app_config.paths.document_chunck_file):
                os.remove(app_config.paths.document_chunck_file)

            return

        dimension = embeddings.shape[1]
        logging.info(f"Création de l'index Faiss optimisé pour la similarité cosinus avec dimension {dimension}...")

        faiss.normalize_L2(embeddings)

        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)
        logging.info(f"Index Faiss créé avec {self.index.ntotal} vecteurs.")

        self._save_index_and_chunks()