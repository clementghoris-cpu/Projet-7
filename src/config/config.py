import os

import yaml
from pydantic import BaseModel, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# --- config.yaml ---

class PathsConfig(BaseModel):
    """Permet de récupérer les différents chemins d'accès des fichiers / dossiers"""
    openagenda_events : str
    faiss_index_file : str
    document_chunck_file : str
    rag_prompt_file : str

    @field_validator("openagenda_events", "faiss_index_file", "document_chunck_file", "rag_prompt_file", mode="after")
    @classmethod
    def make_absolute_path(cls, value : str) -> str:
        if not os.path.isabs(value):
            return os.path.join(os.getcwd(), value)

        return value

class ModelsConfig(BaseModel):
    """Récupère les différents modèles"""
    embeddings_model : str
    llm_model : str
    evaluation_embeddings_model : str
    evaluation_chat_model : str

class IndexerConfig(BaseModel):
    chunk_size : int
    chunk_overlap : int
    embedding_batch_size : int
    search_k : int

class AppConfig(BaseModel):
    """Classe contenant les différentes configurations du projet"""
    paths : PathsConfig    
    models : ModelsConfig
    indexer : IndexerConfig

def load_config(config_file : str = "config.yaml") -> AppConfig:
    """Chargement des données de configuration du projet"""
    with open(config_file, "r", encoding="utf-8") as file:
        raw_data = yaml.safe_load(file)

    return AppConfig.model_validate(raw_data)

# ---------------------------

# --- .env ---

class ApiKeysConfig(BaseSettings):
    """Récupère les configurations de l'accès à mitral dans le fichier .env"""
    model_config = SettingsConfigDict(env_file=".env", env_prefix="api_key_", extra="ignore")
    mistral : SecretStr = "API KEY MISSING"
    gemini : SecretStr = "API KEY MISSING"

# ---------------------------

# --- Variables configuration globale ---
app_config = load_config(config_file = os.path.join(os.getcwd(), "src", "config", "config.yaml"))
api_keys_config = ApiKeysConfig()