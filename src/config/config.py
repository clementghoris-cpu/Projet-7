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
    similarity_threshold : float

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
    """Récupère les clés API dans le fichier .env"""
    model_config = SettingsConfigDict(env_file=".env", env_prefix="api_key_", extra="ignore")
    mistral : SecretStr = "API KEY MISSING"

class ApiSettings(BaseSettings):
    """Récupère les informations de l'API dans le fichier .env"""
    model_config = SettingsConfigDict(env_file=".env", env_prefix="rag_", extra="ignore")
    api_url : str = "http://localhost"
    api_port : int = 8000

class OpenagendaSettings(BaseSettings):
    """Récupère les informations de l'API openagenda dans le fichier .env"""
    model_config = SettingsConfigDict(env_file=".env", env_prefix="openagenda_", extra="ignore")
    api_url : str = "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda/records"
    location_filter : str = "location_city='Lille'"
    start_date : str = "2026-01-01"
    end_date : str = "2026-12-31"
    max_events : int = 1000

# ---------------------------

# --- Variables configuration globale ---
app_config = load_config(config_file = os.path.join(os.getcwd(), "src", "config", "config.yaml"))
api_keys_config = ApiKeysConfig()
api_settings = ApiSettings()
openagenda_settings = OpenagendaSettings()