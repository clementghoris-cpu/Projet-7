from typing import Any

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., description="La question posée par l'utilisateur", example="Quels sont les concerts de jazz prévus ?")

class SourceMetadata(BaseModel):
    uid: str | None = None
    title: str | None = None
    city: str | None = None
    daterange: str | None = None
    canonicalurl: str | None = None

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[dict[str, Any]]
    context : list

class RebuildResponse(BaseModel):
    status: str
    message: str

class HealthResponse(BaseModel):
    status: str
    vector_store_loaded: bool
    total_vectors: int

class MetadataResponse(BaseModel):
    total_chunks: int
    embeddings_model: str
    llm_model: str
    chunk_size: int
    chunk_overlap: int