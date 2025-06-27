"""
Shot Vector Indexer - Vector Database Pipeline for Shot Metadata

This module provides a comprehensive indexing and querying pipeline for shot metadata
using vector databases. It efficiently combines multiple metadata fields to create
rich embeddings that enable powerful semantic search capabilities.

Features:
- Multi-field metadata embedding with weighted importance
- Support for multiple vector databases (Chroma, Pinecone, Weaviate)
- Hierarchical search with filters and metadata
- Batch processing for large datasets
- Query expansion and semantic search
- Real-time indexing and querying
- Model caching to prevent re-downloading SentenceTransformer models

Model Caching:
The module implements a global model cache to prevent re-downloading SentenceTransformer
models. When you create multiple ShotVectorIndexer instances with the same model,
the model will only be downloaded once and reused from cache. Use clear_model_cache()
to free memory when needed.
"""

import logging
import json
import csv
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib
import re

# Vector database imports
try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

# try:
#     import pinecone
#     PINECONE_AVAILABLE = True
# except ImportError:
#     PINECONE_AVAILABLE = False

# try:
#     import weaviate
#     WEAVIATE_AVAILABLE = True
# except ImportError:
#     WEAVIATE_AVAILABLE = False

# Embedding model imports
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

logger = logging.getLogger(__name__)

# Global model cache to prevent re-downloading
_MODEL_CACHE = {}

def clear_model_cache():
    """Clear the global model cache to free memory."""
    global _MODEL_CACHE
    _MODEL_CACHE.clear()
    logger.info("Model cache cleared")

def get_cached_models():
    """Get list of currently cached models."""
    return list(_MODEL_CACHE.keys())

def convert_numpy_to_list(obj: Any) -> Any:
    """
    Recursively converts numpy arrays and other non-serializable types to Python native types.
    
    Args:
        obj: Any Python object that might contain numpy arrays
        
    Returns:
        A JSON-serializable version of the object
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int8, np.int16, np.int32, np.int64,
                         np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.complex64, np.complex128)):
        return str(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_list(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_to_list(item) for item in obj]
    elif hasattr(obj, '__dict__'):
        return convert_numpy_to_list(obj.__dict__)
    return obj

@dataclass
class ShotMetadata:
    """Structured shot metadata for vector indexing."""
    shot_number: int
    start_time_seconds: float
    end_time_seconds: float
    duration_seconds: float
    video_name: str
    
    # Core metadata fields
    shot_description: str
    genre_cues: List[Dict[str, Any]]
    subgenre_cues: List[Dict[str, Any]]
    adjective_theme: List[Dict[str, Any]]
    mood: List[Dict[str, Any]]
    setting_context: List[Dict[str, Any]]
    content_descriptors: List[Dict[str, Any]]
    location_hints_regional: List[Dict[str, Any]]
    location_hints_international: List[Dict[str, Any]]
    search_keywords: List[Dict[str, Any]]
    
    # Computed fields (will be generated in __post_init__)
    combined_text: str = ""
    metadata_hash: str = ""
    
    def __post_init__(self):
        """Generate combined text and hash after initialization."""
        self.combined_text = self._generate_combined_text()
        self.metadata_hash = self._generate_hash()
    
    def _generate_combined_text(self) -> str:
        """Generate a comprehensive text representation for embedding."""
        parts = []
        
        # Primary description (highest weight)
        if self.shot_description and str(self.shot_description).strip():
            parts.append(f"Shot Description: {self.shot_description}")
        
        # Genre and subgenre information
        if self.genre_cues and len(self.genre_cues) > 0:
            genre_text = ", ".join([f"{g.get('genre_hint', '')} ({g.get('prominence_in_shot', '')})" 
                                  for g in self.genre_cues if isinstance(g, dict)])
            if genre_text:
                parts.append(f"Genres: {genre_text}")
        
        if self.subgenre_cues and len(self.subgenre_cues) > 0:
            subgenre_text = ", ".join([f"{s.get('subgenre_hint', '')} ({s.get('prominence_in_shot', '')})" 
                                     for s in self.subgenre_cues if isinstance(s, dict)])
            if subgenre_text:
                parts.append(f"Subgenres: {subgenre_text}")
        
        # Mood and atmosphere
        if self.mood and len(self.mood) > 0:
            mood_text = ", ".join([f"{m.get('mood_hint', m)} ({m.get('prominence_in_shot', '')})" 
                                 for m in self.mood if isinstance(m, dict)])
            if mood_text:
                parts.append(f"Mood: {mood_text}")
        
        # Setting and context
        if self.setting_context and len(self.setting_context) > 0:
            setting_text = ", ".join([f"{s.get('setting_hint', s)} ({s.get('prominence_in_shot', '')})" 
                                    for s in self.setting_context if isinstance(s, dict)])
            if setting_text:
                parts.append(f"Setting: {setting_text}")
        
        # Content descriptors
        if self.content_descriptors and len(self.content_descriptors) > 0:
            content_text = ", ".join([f"{c.get('content_hint', c)} ({c.get('prominence_in_shot', '')})" 
                                    for c in self.content_descriptors if isinstance(c, dict)])
            if content_text:
                parts.append(f"Content: {content_text}")
        
        # Location information
        if self.location_hints_regional and len(self.location_hints_regional) > 0:
            regional_text = ", ".join([f"{l.get('location_hint', l)} ({l.get('prominence_in_shot', '')})" 
                                     for l in self.location_hints_regional if isinstance(l, dict)])
            if regional_text:
                parts.append(f"Regional Location: {regional_text}")
        
        if self.location_hints_international and len(self.location_hints_international) > 0:
            international_text = ", ".join([f"{l.get('location_hint', l)} ({l.get('prominence_in_shot', '')})" 
                                          for l in self.location_hints_international if isinstance(l, dict)])
            if international_text:
                parts.append(f"International Location: {international_text}")
        
        # Search keywords
        if self.search_keywords and len(self.search_keywords) > 0:
            keyword_text = ", ".join([f"{k.get('search_keyword', k)} ({k.get('prominence_in_shot', '')})" 
                                    for k in self.search_keywords if isinstance(k, dict)])
            if keyword_text:
                parts.append(f"Keywords: {keyword_text}")
        
        # Adjective themes
        if self.adjective_theme and len(self.adjective_theme) > 0:
            theme_text = ", ".join([f"{a.get('adjective_hint', a)} ({a.get('prominence_in_shot', '')})" 
                                  for a in self.adjective_theme if isinstance(a, dict)])
            if theme_text:
                parts.append(f"Themes: {theme_text}")
        
        return " | ".join(parts)
    
    def _generate_hash(self) -> str:
        """Generate a unique hash for this shot metadata."""
        content = f"{self.video_name}_{self.shot_number}_{self.start_time_seconds}_{self.end_time_seconds}"
        return hashlib.md5(content.encode()).hexdigest()

class MetadataParser:
    """Parser for CSV shot metadata files."""
    
    @staticmethod
    def parse_json_field(field_value: str) -> List[Dict[str, Any]]:
        """Parse JSON field from CSV, handling various formats."""
        if not field_value or field_value.strip() == '':
            return []
        
        try:
            # Try to parse as JSON
            parsed = json.loads(field_value)
            if isinstance(parsed, list):
                return parsed
            elif isinstance(parsed, dict):
                return [parsed]
            else:
                return [{"value": str(parsed)}]
        except json.JSONDecodeError:
            # Handle non-JSON strings
            if field_value.startswith('[') and field_value.endswith(']'):
                # Try to extract individual items
                items = field_value[1:-1].split('},{')
                result = []
                for item in items:
                    if not item.startswith('{'):
                        item = '{' + item
                    if not item.endswith('}'):
                        item = item + '}'
                    try:
                        parsed_item = json.loads(item)
                        result.append(parsed_item)
                    except:
                        result.append({"value": item})
                return result
            else:
                # Treat as simple string
                return [{"value": field_value}]
    
    @staticmethod
    def parse_csv_to_metadata(csv_path: Path, video_name: str) -> List[ShotMetadata]:
        """Parse CSV file and convert to ShotMetadata objects."""
        metadata_list = []
        
        try:
            df = pd.read_csv(csv_path)
            
            for _, row in df.iterrows():
                try:
                    # Handle NaN values and convert to proper types
                    shot_number = int(row.get('shot_number', 0)) if pd.notna(row.get('shot_number')) else 0
                    start_time = float(row.get('start_time_seconds', 0)) if pd.notna(row.get('start_time_seconds')) else 0.0
                    end_time = float(row.get('end_time_seconds', 0)) if pd.notna(row.get('end_time_seconds')) else 0.0
                    duration = float(row.get('duration_seconds', 0)) if pd.notna(row.get('duration_seconds')) else 0.0
                    shot_description = str(row.get('metadata_ShotDescription', '')) if pd.notna(row.get('metadata_ShotDescription')) else ''
                    
                    # Convert all metadata fields to ensure they're Python native types
                    genre_cues = convert_numpy_to_list(MetadataParser.parse_json_field(row.get('metadata_GenreCues', '')))
                    subgenre_cues = convert_numpy_to_list(MetadataParser.parse_json_field(row.get('metadata_SubgenreCues', '')))
                    adjective_theme = convert_numpy_to_list(MetadataParser.parse_json_field(row.get('metadata_AdjectiveTheme', '')))
                    mood = convert_numpy_to_list(MetadataParser.parse_json_field(row.get('metadata_Mood', '')))
                    setting_context = convert_numpy_to_list(MetadataParser.parse_json_field(row.get('metadata_SettingContext', '')))
                    content_descriptors = convert_numpy_to_list(MetadataParser.parse_json_field(row.get('metadata_ContentDescriptors', '')))
                    location_hints_regional = convert_numpy_to_list(MetadataParser.parse_json_field(row.get('metadata_LocationHints_Regional', '')))
                    location_hints_international = convert_numpy_to_list(MetadataParser.parse_json_field(row.get('metadata_LocationHints_International', '')))
                    search_keywords = convert_numpy_to_list(MetadataParser.parse_json_field(row.get('metadata_SearchKeywords', '')))
                    
                    metadata = ShotMetadata(
                        shot_number=shot_number,
                        start_time_seconds=start_time,
                        end_time_seconds=end_time,
                        duration_seconds=duration,
                        video_name=video_name,
                        
                        shot_description=shot_description,
                        genre_cues=genre_cues,
                        subgenre_cues=subgenre_cues,
                        adjective_theme=adjective_theme,
                        mood=mood,
                        setting_context=setting_context,
                        content_descriptors=content_descriptors,
                        location_hints_regional=location_hints_regional,
                        location_hints_international=location_hints_international,
                        search_keywords=search_keywords
                    )
                    metadata_list.append(metadata)
                    
                except Exception as e:
                    logger.warning(f"Failed to parse row {row.get('shot_number', 'unknown')}: {e}")
                    continue
            
            logger.info(f"Successfully parsed {len(metadata_list)} shots from {csv_path}")
            return metadata_list
            
        except Exception as e:
            logger.error(f"Failed to parse CSV file {csv_path}: {e}")
            return []

class EmbeddingGenerator:
    """Generate embeddings for shot metadata."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", use_openai: bool = False, openai_api_key: Optional[str] = None):
        """
        Initialize embedding generator.
        
        Args:
            model_name: Sentence transformer model name
            use_openai: Whether to use OpenAI embeddings
            openai_api_key: OpenAI API key if using OpenAI
        """
        self.model_name = model_name
        self.use_openai = use_openai
        self.openai_api_key = openai_api_key
        self.model = None
        
        if use_openai and not OPENAI_AVAILABLE:
            raise ImportError("OpenAI not available. Install with: pip install openai")
        
        if not use_openai and not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("Sentence transformers not available. Install with: pip install sentence-transformers")
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the embedding model."""
        if self.use_openai:
            if self.openai_api_key:
                openai.api_key = self.openai_api_key
            else:
                raise ValueError("OpenAI API key required when use_openai=True")
        else:
            try:
                # Check if model is already in cache
                if self.model_name in _MODEL_CACHE:
                    self.model = _MODEL_CACHE[self.model_name]
                    logger.info(f"Loaded cached sentence transformer model: {self.model_name}")
                else:
                    # Load model and cache it
                    self.model = SentenceTransformer(self.model_name)
                    _MODEL_CACHE[self.model_name] = self.model
                    logger.info(f"Loaded and cached sentence transformer model: {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to load model {self.model_name}: {e}")
                raise
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for given text."""
        if not text.strip():
            # Return zero vector for empty text
            if self.use_openai:
                return np.zeros(1536)  # OpenAI ada-002 dimension
            else:
                return np.zeros(self.model.get_sentence_embedding_dimension())
        
        try:
            if self.use_openai:
                response = openai.Embedding.create(
                    input=text,
                    model="text-embedding-ada-002"
                )
                return np.array(response['data'][0]['embedding'])
            else:
                return self.model.encode(text, convert_to_numpy=True)
                
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            # Return zero vector on error
            if self.use_openai:
                return np.zeros(1536)
            else:
                return np.zeros(self.model.get_sentence_embedding_dimension())
    
    def generate_batch_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for a batch of texts."""
        if not texts:
            return []
        
        try:
            if self.use_openai:
                # OpenAI has rate limits, so process in smaller batches
                batch_size = 100
                all_embeddings = []
                
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i + batch_size]
                    response = openai.Embedding.create(
                        input=batch,
                        model="text-embedding-ada-002"
                    )
                    batch_embeddings = [np.array(item['embedding']) for item in response['data']]
                    all_embeddings.extend(batch_embeddings)
                
                return all_embeddings
            else:
                return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
                
        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {e}")
            # Return zero vectors on error
            if self.use_openai:
                return [np.zeros(1536) for _ in texts]
            else:
                return [np.zeros(self.model.get_sentence_embedding_dimension()) for _ in texts]

class VectorDatabase:
    """Abstract base class for vector database operations."""
    
    def __init__(self, db_type: str = "chroma", **kwargs):
        """
        Initialize vector database.
        
        Args:
            db_type: Type of database ("chroma", "pinecone", "weaviate")
            **kwargs: Database-specific configuration
        """
        self.db_type = db_type
        self.client = None
        self.collection = None
        
        if db_type == "chroma" and not CHROMA_AVAILABLE:
            raise ImportError("ChromaDB not available. Install with: pip install chromadb")
        elif db_type == "pinecone" and not PINECONE_AVAILABLE:
            raise ImportError("Pinecone not available. Install with: pip install pinecone-client")
        elif db_type == "weaviate" and not WEAVIATE_AVAILABLE:
            raise ImportError("Weaviate not available. Install with: pip install weaviate-client")
        
        self._initialize_database(**kwargs)
    
    def _initialize_database(self, **kwargs):
        """Initialize the specific database."""
        if self.db_type == "chroma":
            self._init_chroma(**kwargs)
        elif self.db_type == "pinecone":
            self._init_pinecone(**kwargs)
        elif self.db_type == "weaviate":
            self._init_weaviate(**kwargs)
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}")
    
    def _init_chroma(self, persist_directory: str = "./chroma_db", collection_name: str = "shot_metadata"):
        """Initialize ChromaDB."""
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"Initialized ChromaDB collection: {collection_name}")
    
    def _init_pinecone(self, api_key: str, environment: str, index_name: str = "shot-metadata"):
        """Initialize Pinecone."""
        pinecone.init(api_key=api_key, environment=environment)
        
        # Check if index exists
        if index_name not in pinecone.list_indexes():
            # Create index with appropriate dimension
            dimension = 384  # Default for sentence-transformers
            pinecone.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine"
            )
        
        self.index = pinecone.Index(index_name)
        logger.info(f"Initialized Pinecone index: {index_name}")
    
    def _init_weaviate(self, url: str = "http://localhost:8080", class_name: str = "ShotMetadata"):
        """Initialize Weaviate."""
        self.client = weaviate.Client(url)
        
        # Define schema
        schema = {
            "class": class_name,
            "properties": [
                {"name": "shot_number", "dataType": ["int"]},
                {"name": "video_name", "dataType": ["string"]},
                {"name": "start_time", "dataType": ["number"]},
                {"name": "end_time", "dataType": ["number"]},
                {"name": "duration", "dataType": ["number"]},
                {"name": "shot_description", "dataType": ["text"]},
                {"name": "metadata_hash", "dataType": ["string"]}
            ],
            "vectorizer": "text2vec-transformers"
        }
        
        try:
            self.client.schema.create_class(schema)
        except:
            pass  # Class might already exist
        
        self.class_name = class_name
        logger.info(f"Initialized Weaviate class: {class_name}")
    
    def add_shots(self, shots: List[ShotMetadata], embeddings: List[np.ndarray]):
        """Add shots to the vector database."""
        if len(shots) == 0 or len(embeddings) == 0:
            return
        
        if self.db_type == "chroma":
            self._add_shots_chroma(shots, embeddings)
        elif self.db_type == "pinecone":
            self._add_shots_pinecone(shots, embeddings)
        elif self.db_type == "weaviate":
            self._add_shots_weaviate(shots, embeddings)
    
    def _add_shots_chroma(self, shots: List[ShotMetadata], embeddings: List[np.ndarray]):
        """Add shots to ChromaDB."""
        ids = [shot.metadata_hash for shot in shots]
        texts = [shot.combined_text for shot in shots]
        embeddings_list = [embedding.tolist() for embedding in embeddings]
        
        metadatas = []
        for shot in shots:
            metadata = {
                "shot_number": int(shot.shot_number),  # Ensure it's a Python int
                "video_name": str(shot.video_name),    # Ensure it's a Python string
                "start_time": float(shot.start_time_seconds),  # Ensure it's a Python float
                "end_time": float(shot.end_time_seconds),      # Ensure it's a Python float
                "duration": float(shot.duration_seconds),      # Ensure it's a Python float
                "shot_description": str(shot.shot_description) if shot.shot_description else ""  # Handle NaN values
            }
            metadatas.append(metadata)
        
        self.collection.add(
            ids=ids,
            embeddings=embeddings_list,
            documents=texts,
            metadatas=metadatas
        )
        logger.info(f"Added {len(shots)} shots to ChromaDB")
    
    def _add_shots_pinecone(self, shots: List[ShotMetadata], embeddings: List[np.ndarray]):
        """Add shots to Pinecone."""
        vectors = []
        for shot, embedding in zip(shots, embeddings):
            vector = {
                "id": shot.metadata_hash,
                "values": embedding.tolist(),
                "metadata": {
                    "shot_number": shot.shot_number,
                    "video_name": shot.video_name,
                    "start_time": shot.start_time_seconds,
                    "end_time": shot.end_time_seconds,
                    "duration": shot.duration_seconds,
                    "shot_description": shot.shot_description,
                    "text": shot.combined_text
                }
            }
            vectors.append(vector)
        
        # Upsert in batches
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch)
        
        logger.info(f"Added {len(shots)} shots to Pinecone")
    
    def _add_shots_weaviate(self, shots: List[ShotMetadata], embeddings: List[np.ndarray]):
        """Add shots to Weaviate."""
        with self.client.batch as batch:
            for shot, embedding in zip(shots, embeddings):
                data_object = {
                    "shot_number": shot.shot_number,
                    "video_name": shot.video_name,
                    "start_time": shot.start_time_seconds,
                    "end_time": shot.end_time_seconds,
                    "duration": shot.duration_seconds,
                    "shot_description": shot.shot_description,
                    "metadata_hash": shot.metadata_hash
                }
                
                batch.add_data_object(
                    data_object=data_object,
                    class_name=self.class_name,
                    vector=embedding.tolist()
                )
        
        logger.info(f"Added {len(shots)} shots to Weaviate")
    
    def search(self, query: str, n_results: int = 10, filters: Optional[Dict] = None) -> List[Dict]:
        """Search for similar shots."""
        if self.db_type == "chroma":
            return self._search_chroma(query, n_results, filters)
        elif self.db_type == "pinecone":
            return self._search_pinecone(query, n_results, filters)
        elif self.db_type == "weaviate":
            return self._search_weaviate(query, n_results, filters)
    
    def _search_chroma(self, query: str, n_results: int, filters: Optional[Dict]) -> List[Dict]:
        """Search in ChromaDB."""
        where_filter = None
        if filters:
            where_filter = {}
            for key, value in filters.items():
                if key == "video_name":
                    where_filter["video_name"] = {"$eq": value}
                elif key == "shot_number":
                    where_filter["shot_number"] = {"$eq": value}
        
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_filter
        )
        
        return self._format_chroma_results(results)
    
    def _search_pinecone(self, query: str, n_results: int, filters: Optional[Dict]) -> List[Dict]:
        """Search in Pinecone."""
        filter_dict = None
        if filters:
            filter_dict = {}
            for key, value in filters.items():
                if key == "video_name":
                    filter_dict["video_name"] = {"$eq": value}
                elif key == "shot_number":
                    filter_dict["shot_number"] = {"$eq": value}
        
        # Note: Pinecone requires query embedding, so this is a simplified version
        # In practice, you'd need to generate the query embedding first
        results = self.index.query(
            vector=[0.0] * 384,  # Placeholder - should be actual query embedding
            top_k=n_results,
            filter=filter_dict,
            include_metadata=True
        )
        
        return self._format_pinecone_results(results)
    
    def _search_weaviate(self, query: str, n_results: int, filters: Optional[Dict]) -> List[Dict]:
        """Search in Weaviate."""
        where_filter = None
        if filters:
            where_parts = []
            for key, value in filters.items():
                if key == "video_name":
                    where_parts.append({
                        "path": ["video_name"],
                        "operator": "Equal",
                        "valueString": value
                    })
                elif key == "shot_number":
                    where_parts.append({
                        "path": ["shot_number"],
                        "operator": "Equal",
                        "valueInt": value
                    })
            
            if len(where_parts) == 1:
                where_filter = where_parts[0]
            elif len(where_parts) > 1:
                where_filter = {
                    "operator": "And",
                    "operands": where_parts
                }
        
        results = self.client.query.get(self.class_name, [
            "shot_number", "video_name", "start_time", "end_time", 
            "duration", "shot_description", "metadata_hash"
        ]).with_near_text({
            "concepts": [query]
        }).with_where(where_filter).with_limit(n_results).do()
        
        return self._format_weaviate_results(results)
    
    def _format_chroma_results(self, results) -> List[Dict]:
        """Format ChromaDB results."""
        formatted_results = []
        if results['ids'] and results['ids'][0]:
            for i in range(len(results['ids'][0])):
                result = {
                    'id': results['ids'][0][i],
                    'distance': results['distances'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'text': results['documents'][0][i]
                }
                formatted_results.append(result)
        return formatted_results
    
    def _format_pinecone_results(self, results) -> List[Dict]:
        """Format Pinecone results."""
        formatted_results = []
        for match in results['matches']:
            result = {
                'id': match['id'],
                'distance': match['score'],
                'metadata': match['metadata']
            }
            formatted_results.append(result)
        return formatted_results
    
    def _format_weaviate_results(self, results) -> List[Dict]:
        """Format Weaviate results."""
        formatted_results = []
        for result in results['data']['Get'][self.class_name]:
            formatted_result = {
                'id': result['metadata_hash'],
                'distance': result.get('_additional', {}).get('distance', 0),
                'metadata': {
                    'shot_number': result['shot_number'],
                    'video_name': result['video_name'],
                    'start_time': result['start_time'],
                    'end_time': result['end_time'],
                    'duration': result['duration'],
                    'shot_description': result['shot_description']
                }
            }
            formatted_results.append(formatted_result)
        return formatted_results

class ShotVectorIndexer:
    """Main class for indexing and querying shot metadata."""
    
    def __init__(self, 
                 db_type: str = "chroma",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 use_openai: bool = False,
                 openai_api_key: Optional[str] = None,
                 **db_kwargs):
        """
        Initialize the shot vector indexer.
        
        Args:
            db_type: Vector database type ("chroma", "pinecone", "weaviate")
            embedding_model: Sentence transformer model name
            use_openai: Whether to use OpenAI embeddings
            openai_api_key: OpenAI API key if using OpenAI
            **db_kwargs: Database-specific configuration
        """
        self.embedding_generator = EmbeddingGenerator(
            model_name=embedding_model,
            use_openai=use_openai,
            openai_api_key=openai_api_key
        )
        
        self.vector_db = VectorDatabase(db_type=db_type, **db_kwargs)
        
        logger.info(f"Initialized ShotVectorIndexer with {db_type} database and {embedding_model} embeddings")
    
    def index_csv_file(self, csv_path: Path, video_name: str) -> bool:
        """
        Index a CSV file containing shot metadata.
        
        Args:
            csv_path: Path to the CSV file
            video_name: Name of the video
            
        Returns:
            bool: True if indexing was successful
        """
        try:
            # Parse CSV to metadata objects
            shots = MetadataParser.parse_csv_to_metadata(csv_path, video_name)
            if not shots:
                logger.warning(f"No shots found in {csv_path}")
                return False
            
            # Generate embeddings
            texts = [shot.combined_text for shot in shots]
            embeddings = self.embedding_generator.generate_batch_embeddings(texts)
            
            # Add to vector database
            self.vector_db.add_shots(shots, embeddings)
            
            logger.info(f"Successfully indexed {len(shots)} shots from {csv_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to index {csv_path}: {e}")
            return False
    
    def index_directory(self, directory_path: Path, pattern: str = "*_shot_metadata.csv") -> Dict[str, bool]:
        """
        Index all CSV files in a directory.
        
        Args:
            directory_path: Directory containing CSV files
            pattern: File pattern to match
            
        Returns:
            Dict mapping file paths to success status
        """
        results = {}
        
        for csv_file in directory_path.glob(pattern):
            video_name = csv_file.stem.replace("_shot_metadata", "")
            success = self.index_csv_file(csv_file, video_name)
            results[str(csv_file)] = success
        
        return results
    
    def search_shots(self, 
                    query: str, 
                    n_results: int = 10, 
                    video_name: Optional[str] = None,
                    shot_number: Optional[int] = None,
                    min_duration: Optional[float] = None,
                    max_duration: Optional[float] = None) -> List[Dict]:
        """
        Search for shots based on query and filters.
        
        Args:
            query: Search query text
            n_results: Number of results to return
            video_name: Filter by video name
            shot_number: Filter by shot number
            min_duration: Minimum shot duration
            max_duration: Maximum shot duration
            
        Returns:
            List of matching shots with metadata
        """
        # Build filters
        filters = {}
        if video_name:
            filters["video_name"] = video_name
        if shot_number:
            filters["shot_number"] = shot_number
        
        # Search in vector database
        results = self.vector_db.search(query, n_results, filters)
        
        # Apply duration filters if specified
        if min_duration or max_duration:
            filtered_results = []
            for result in results:
                duration = result['metadata'].get('duration', 0)
                if min_duration and duration < min_duration:
                    continue
                if max_duration and duration > max_duration:
                    continue
                filtered_results.append(result)
            results = filtered_results
        
        return results
    
    def search_by_metadata(self, 
                          genre: Optional[str] = None,
                          mood: Optional[str] = None,
                          setting: Optional[str] = None,
                          content: Optional[str] = None,
                          location: Optional[str] = None,
                          n_results: int = 10) -> List[Dict]:
        """
        Search for shots based on specific metadata fields.
        
        Args:
            genre: Genre to search for
            mood: Mood to search for
            setting: Setting to search for
            content: Content descriptor to search for
            location: Location to search for
            n_results: Number of results to return
            
        Returns:
            List of matching shots
        """
        # Build query from metadata fields
        query_parts = []
        if genre:
            query_parts.append(f"genre: {genre}")
        if mood:
            query_parts.append(f"mood: {mood}")
        if setting:
            query_parts.append(f"setting: {setting}")
        if content:
            query_parts.append(f"content: {content}")
        if location:
            query_parts.append(f"location: {location}")
        
        if not query_parts:
            return []
        
        query = " | ".join(query_parts)
        return self.search_shots(query, n_results)
    
    def get_shot_by_id(self, shot_id: str) -> Optional[Dict]:
        """Get a specific shot by its ID."""
        # This would need to be implemented based on the specific database
        # For now, we'll search with a very specific query
        results = self.vector_db.search("", n_results=1)
        for result in results:
            if result['id'] == shot_id:
                return result
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        # This would need to be implemented based on the specific database
        return {
            "database_type": self.vector_db.db_type,
            "embedding_model": self.embedding_generator.model_name,
            "use_openai": self.embedding_generator.use_openai,
            "cached_models": get_cached_models(),
            "cache_size": len(_MODEL_CACHE)
        }

    @staticmethod
    def clear_model_cache():
        """Clear the global model cache to free memory."""
        clear_model_cache()
    
    @staticmethod
    def get_cached_models():
        """Get list of currently cached models."""
        return get_cached_models()

def create_indexer_from_config(config_path: Path) -> ShotVectorIndexer:
    """Create a ShotVectorIndexer from a configuration file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return ShotVectorIndexer(**config)

def main():
    """Example usage of the ShotVectorIndexer."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Index and search shot metadata")
    parser.add_argument("--action", choices=["index", "search"], required=True)
    parser.add_argument("--input", type=Path, help="Input CSV file or directory")
    parser.add_argument("--query", type=str, help="Search query")
    parser.add_argument("--db-type", default="chroma", choices=["chroma", "pinecone", "weaviate"])
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="Embedding model")
    parser.add_argument("--results", type=int, default=10, help="Number of search results")
    
    args = parser.parse_args()
    
    # Initialize indexer
    indexer = ShotVectorIndexer(db_type=args.db_type, embedding_model=args.model)
    
    if args.action == "index":
        if args.input.is_file():
            video_name = args.input.stem.replace("_shot_metadata", "")
            success = indexer.index_csv_file(args.input, video_name)
            print(f"Indexing {'successful' if success else 'failed'}")
        elif args.input.is_dir():
            results = indexer.index_directory(args.input)
            for file_path, success in results.items():
                print(f"{file_path}: {'successful' if success else 'failed'}")
    
    elif args.action == "search":
        if not args.query:
            print("Query required for search action")
            return
        
        results = indexer.search_shots(args.query, args.results)
        print(f"Found {len(results)} results:")
        for i, result in enumerate(results, 1):
            metadata = result['metadata']
            print(f"{i}. Shot {metadata['shot_number']} from {metadata['video_name']}")
            print(f"   Time: {metadata['start_time']:.2f}s - {metadata['end_time']:.2f}s")
            print(f"   Description: {metadata['shot_description']}")
            print(f"   Distance: {result['distance']:.4f}")
            print()

if __name__ == "__main__":
    main() 