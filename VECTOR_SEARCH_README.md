# Shot Metadata Vector Search System

A comprehensive vector database indexing and querying pipeline for shot metadata that enables powerful semantic search capabilities across video shots.

## 🚀 Features

### Core Functionality
- **Multi-field Metadata Embedding**: Combines all shot metadata fields (description, genre, mood, setting, etc.) into rich text representations
- **Multiple Vector Database Support**: ChromaDB (local), Pinecone (cloud), and Weaviate (optional)
- **Flexible Embedding Models**: Sentence Transformers and OpenAI embeddings
- **Advanced Search Capabilities**: Natural language queries with metadata filtering
- **Batch Processing**: Efficient indexing of large datasets
- **Real-time Querying**: Fast semantic search with similarity scoring

### Search Capabilities
- **Natural Language Queries**: "happy family scene", "bathroom hair styling", "sad emotional moment"
- **Metadata-based Filtering**: Filter by video name, shot number, duration, etc.
- **Semantic Similarity**: Find shots with similar content, mood, or setting
- **Multi-modal Search**: Combine visual descriptions with metadata attributes

## 📊 Metadata Fields Supported

The system processes and indexes the following shot metadata fields:

| Field | Description | Example |
|-------|-------------|---------|
| `shot_description` | Primary visual description | "A young girl with curly hair sitting in a bathtub" |
| `genre_cues` | Genre hints with prominence scores | `[{"genre_hint": "Family", "prominence_in_shot": 1.0}]` |
| `subgenre_cues` | Subgenre hints | `[{"subgenre_hint": "Children's", "prominence_in_shot": 1.0}]` |
| `adjective_theme` | Descriptive themes | `[{"adjective_hint": "happy", "prominence_in_shot": 0.8}]` |
| `mood` | Emotional atmosphere | `[{"mood_hint": "calm", "prominence_in_shot": 0.9}]` |
| `setting_context` | Location/setting info | `[{"setting_hint": "home", "prominence_in_shot": 1.0}]` |
| `content_descriptors` | Content elements | `[{"content_hint": "hair care", "prominence_in_shot": 0.7}]` |
| `location_hints_regional` | Regional locations | `[{"location_hint": "New York", "prominence_in_shot": 0.5}]` |
| `location_hints_international` | International locations | `[{"location_hint": "London", "prominence_in_shot": 0.3}]` |
| `search_keywords` | Searchable keywords | `[{"search_keyword": "family", "prominence_in_shot": 1.0}]` |

## 🛠️ Installation

### 1. Install Dependencies

```bash
pip install -r requirements_vector_search.txt
```

### 2. Optional: Set up OpenAI (for OpenAI embeddings)

```bash
export OPENAI_API_KEY="your-api-key-here"
```

## 📖 Usage

### Basic Usage

```python
from pathlib import Path
from src.videoqa.shot_vector_indexer import ShotVectorIndexer

# Initialize indexer
indexer = ShotVectorIndexer(
    db_type="chroma",  # or "pinecone"
    embedding_model="all-MiniLM-L6-v2"
)

# Index a CSV file
csv_path = Path("processed_videos_output_module2_2_scenes/Hair Love/Hair Love_shot_metadata.csv")
success = indexer.index_csv_file(csv_path, "Hair Love")

# Search for shots
results = indexer.search_shots("happy family scene", n_results=5)
for result in results:
    print(f"Shot {result['metadata']['shot_number']}: {result['metadata']['shot_description']}")
```

### Command Line Demo

```bash
# Index shot metadata
python scripts/demo_shot_vector_search.py --action index --input processed_videos_output_module2_2_scenes/

# Search for shots
python scripts/demo_shot_vector_search.py --action search --query "happy family scene"

# Run full demo
python scripts/demo_shot_vector_search.py --action demo --input processed_videos_output_module2_2_scenes/

# See query examples
python scripts/demo_shot_vector_search.py --action examples
```

## 🔍 Search Examples

### Natural Language Queries
```python
# Find family scenes
results = indexer.search_shots("happy family scene")

# Find hair care scenes
results = indexer.search_shots("bathroom hair styling")

# Find emotional moments
results = indexer.search_shots("sad emotional moment")

# Find specific objects
results = indexer.search_shots("cat in the scene")
```

### Metadata-based Search
```python
# Search by specific metadata fields
results = indexer.search_by_metadata(
    genre="Family",
    mood="happy",
    setting="home"
)
```

### Advanced Filtering
```python
# Search with filters
results = indexer.search_shots(
    query="family scene",
    n_results=10,
    video_name="Hair Love",
    min_duration=5.0,
    max_duration=30.0
)
```

## 🏗️ Architecture

### Components

1. **ShotMetadata**: Structured data class for shot metadata
2. **MetadataParser**: Parses CSV files and converts to metadata objects
3. **EmbeddingGenerator**: Generates embeddings using various models
4. **VectorDatabase**: Abstract interface for different vector databases
5. **ShotVectorIndexer**: Main class orchestrating indexing and search

### Data Flow

```
CSV File → MetadataParser → ShotMetadata → EmbeddingGenerator → VectorDatabase
                                                                    ↓
Query → EmbeddingGenerator → VectorDatabase → Search Results
```

### Text Generation Strategy

The system creates rich text representations by combining metadata fields:

```
"Shot Description: A young girl with curly hair sitting in a bathtub | 
 Genres: Family (1.0), Children's (1.0) | 
 Mood: happy (0.8), calm (0.9) | 
 Setting: home (1.0), bathroom (1.0) | 
 Content: hair care (0.7), personal grooming (0.8) | 
 Keywords: family, home, happy, calm"
```

## 🗄️ Vector Database Options

### ChromaDB (Recommended for Local Use)
- **Pros**: Local storage, easy setup, good performance
- **Cons**: Limited scalability
- **Use Case**: Development, small to medium datasets

```python
indexer = ShotVectorIndexer(
    db_type="chroma",
    persist_directory="./chroma_db",
    collection_name="shot_metadata"
)
```

### Pinecone (Recommended for Production)
- **Pros**: Cloud-based, highly scalable, production-ready
- **Cons**: Requires API key, costs money
- **Use Case**: Production systems, large datasets

```python
indexer = ShotVectorIndexer(
    db_type="pinecone",
    api_key="your-pinecone-api-key",
    environment="us-west1-gcp",
    index_name="shot-metadata"
)
```

## 🎯 Query Optimization

### Best Practices

1. **Use Specific Queries**: "happy family scene" vs "good"
2. **Combine Multiple Concepts**: "bathroom hair styling" vs "bathroom"
3. **Leverage Metadata Filters**: Use video_name, duration filters
4. **Consider Context**: Include relevant objects, actions, emotions

### Query Examples by Category

#### Emotional Queries
- "happy family moment"
- "sad emotional scene"
- "excited child"
- "calm peaceful atmosphere"

#### Action Queries
- "hair styling routine"
- "family interaction"
- "child playing"
- "parent helping child"

#### Setting Queries
- "bathroom scene"
- "bedroom setting"
- "home environment"
- "hospital room"

#### Object Queries
- "cat in scene"
- "tablet or laptop"
- "mirror reflection"
- "pink dress"

## 📈 Performance Considerations

### Indexing Performance
- **Batch Processing**: Process multiple files efficiently
- **Embedding Caching**: Reuse embeddings when possible
- **Memory Management**: Process large datasets in chunks

### Search Performance
- **Query Optimization**: Use specific, descriptive queries
- **Result Limiting**: Limit results to improve speed
- **Filtering**: Use metadata filters to narrow search space

### Scaling Considerations
- **Database Choice**: ChromaDB for local, Pinecone for cloud
- **Embedding Model**: Choose based on speed vs accuracy trade-off
- **Batch Size**: Optimize batch size for your hardware

## 🔧 Configuration

### Embedding Models

| Model | Speed | Quality | Size | Use Case |
|-------|-------|---------|------|----------|
| `all-MiniLM-L6-v2` | Fast | Good | Small | Development |
| `all-mpnet-base-v2` | Medium | Better | Medium | Production |
| `text-embedding-ada-002` | Slow | Best | Large | High-quality |

### Database Configuration

```python
# ChromaDB
chroma_config = {
    "persist_directory": "./chroma_db",
    "collection_name": "shot_metadata"
}

# Pinecone
pinecone_config = {
    "api_key": "your-api-key",
    "environment": "us-west1-gcp",
    "index_name": "shot-metadata"
}
```

## 🧪 Testing

### Run Tests
```bash
pytest tests/test_shot_vector_indexer.py -v
```

### Test Coverage
```bash
pytest tests/test_shot_vector_indexer.py --cov=src/videoqa/shot_vector_indexer
```

## 📝 API Reference

### ShotVectorIndexer

#### Methods

- `index_csv_file(csv_path, video_name)`: Index a single CSV file
- `index_directory(directory_path, pattern)`: Index all CSV files in directory
- `search_shots(query, n_results, **filters)`: Search for shots
- `search_by_metadata(**metadata_filters)`: Search by specific metadata
- `get_shot_by_id(shot_id)`: Get specific shot by ID
- `get_statistics()`: Get database statistics

#### Parameters

- `db_type`: "chroma", "pinecone", or "weaviate"
- `embedding_model`: Sentence transformer model name
- `use_openai`: Whether to use OpenAI embeddings
- `openai_api_key`: OpenAI API key

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Troubleshooting

### Common Issues

1. **Import Errors**: Install missing dependencies
2. **Memory Issues**: Reduce batch size or use smaller embedding model
3. **Database Connection**: Check API keys and network connectivity
4. **Performance Issues**: Optimize queries and use appropriate filters

### Getting Help

- Check the demo script for usage examples
- Review the test files for implementation details
- Open an issue for bugs or feature requests 