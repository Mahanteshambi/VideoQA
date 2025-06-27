# Shot Metadata Vector Search Solution

## 🎯 Problem Analysis

You have CSV files containing rich shot metadata extracted by LLMs, including:
- `metadata_ShotDescription`: Visual descriptions of shots
- `metadata_GenreCues`: Genre hints with prominence scores
- `metadata_SubgenreCues`: Subgenre information
- `metadata_AdjectiveTheme`: Descriptive themes
- `metadata_Mood`: Emotional atmosphere
- `metadata_SettingContext`: Location/setting information
- `metadata_ContentDescriptors`: Content elements
- `metadata_LocationHints_Regional/International`: Geographic information
- `metadata_SearchKeywords`: Searchable keywords

You need an efficient way to:
1. **Index** this metadata in a vector database
2. **Query** shots using natural language
3. **Search** across multiple metadata dimensions
4. **Filter** results by various criteria

## 🏗️ Solution Architecture

### 1. Multi-Field Text Generation Strategy

The key innovation is combining all metadata fields into a rich, structured text representation that preserves context and relationships.

### 2. Flexible Vector Database Support

- **ChromaDB** (Local): Easy setup, good for development
- **Pinecone** (Cloud): Scalable, production-ready
- **Weaviate** (Optional): Advanced features

### 3. Multiple Embedding Models

- **Sentence Transformers**: Fast, local processing
- **OpenAI Embeddings**: High quality, cloud-based

## 🚀 Key Features

### 1. Intelligent Metadata Parsing
- Robust JSON handling for various formats
- Graceful fallback for malformed data
- Batch processing for large datasets

### 2. Advanced Search Capabilities
- Natural language queries
- Metadata-based filtering
- Advanced filtering by duration, video, etc.

### 3. Production Ready
- Comprehensive error handling
- Extensive documentation
- Easy integration

## 🛠️ Implementation Files

1. **`src/videoqa/shot_vector_indexer.py`**: Main implementation
2. **`scripts/demo_shot_vector_search.py`**: Demo script
3. **`test_vector_search.py`**: Test suite
4. **`requirements_vector_search.txt`**: Dependencies
5. **`VECTOR_SEARCH_README.md`**: Documentation

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements_vector_search.txt

# Run test
python test_vector_search.py

# Try demo
python scripts/demo_shot_vector_search.py --action demo --input processed_videos_output_module2_2_scenes/
```

## 🎯 Benefits

1. **Rich Context Preservation**: Combines all metadata fields meaningfully
2. **Flexible Querying**: Natural language and metadata-based search
3. **Scalable Architecture**: Multiple database support
4. **Production Ready**: Comprehensive error handling
5. **Easy Integration**: Simple API and command-line tools 