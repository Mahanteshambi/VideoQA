#!/usr/bin/env python3
"""
Demo script for Shot Vector Indexer

This script demonstrates how to:
1. Index shot metadata from CSV files
2. Search for shots using natural language queries
3. Filter results by various criteria
4. Perform metadata-based searches

Usage:
    python demo_shot_vector_search.py --action index --input path/to/csv/files
    python demo_shot_vector_search.py --action search --query "happy family scene"
"""

import argparse
import logging
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from videoqa.shot_vector_indexer import ShotVectorIndexer, MetadataParser

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def demo_indexing(indexer: ShotVectorIndexer, input_path: Path):
    """Demonstrate indexing functionality."""
    print("=== Shot Metadata Indexing Demo ===")
    
    if input_path.is_file():
        # Index single file
        video_name = input_path.stem.replace("_shot_metadata", "")
        print(f"Indexing single file: {input_path}")
        print(f"Video name: {video_name}")
        
        success = indexer.index_csv_file(input_path, video_name)
        if success:
            print("✅ Indexing successful!")
        else:
            print("❌ Indexing failed!")
            
    elif input_path.is_dir():
        # Index directory
        print(f"Indexing directory: {input_path}")
        results = indexer.index_directory(input_path)
        
        print("\nIndexing Results:")
        for file_path, success in results.items():
            status = "✅ Success" if success else "❌ Failed"
            print(f"  {Path(file_path).name}: {status}")
        
        success_count = sum(1 for success in results.values() if success)
        total_count = len(results)
        print(f"\nSummary: {success_count}/{total_count} files indexed successfully")
        
    else:
        print(f"❌ Invalid input path: {input_path}")

def demo_searching(indexer: ShotVectorIndexer, query: str, n_results: int = 5):
    """Demonstrate searching functionality."""
    print("=== Shot Metadata Search Demo ===")
    print(f"Query: '{query}'")
    print(f"Max results: {n_results}")
    
    # Perform search
    results = indexer.search_shots(query, n_results)
    
    if not results:
        print("❌ No results found")
        return
    
    print(f"\nFound {len(results)} results:")
    print("-" * 80)
    
    for i, result in enumerate(results, 1):
        metadata = result['metadata']
        distance = result['distance']
        
        print(f"{i}. Shot {metadata['shot_number']} from '{metadata['video_name']}'")
        print(f"   Time: {metadata['start_time']:.2f}s - {metadata['end_time']:.2f}s")
        print(f"   Duration: {metadata['duration']:.2f}s")
        print(f"   Similarity Score: {distance:.4f}")
        
        if metadata.get('shot_description'):
            desc = metadata['shot_description']
            if len(desc) > 100:
                desc = desc[:100] + "..."
            print(f"   Description: {desc}")
        
        print()

def demo_metadata_search(indexer: ShotVectorIndexer):
    """Demonstrate metadata-based searching."""
    print("=== Metadata-Based Search Demo ===")
    
    # Example searches
    search_examples = [
        {
            "name": "Family scenes",
            "params": {"genre": "Family", "mood": "happy"}
        },
        {
            "name": "Hair care scenes",
            "params": {"content": "hair styling", "setting": "bathroom"}
        },
        {
            "name": "Sad emotional scenes",
            "params": {"mood": "sad", "content": "emotional"}
        },
        {
            "name": "Home settings",
            "params": {"setting": "home", "location": "home"}
        }
    ]
    
    for example in search_examples:
        print(f"\n--- {example['name']} ---")
        results = indexer.search_by_metadata(**example['params'], n_results=3)
        
        if results:
            for i, result in enumerate(results, 1):
                metadata = result['metadata']
                print(f"  {i}. Shot {metadata['shot_number']} from '{metadata['video_name']}' "
                      f"({metadata['start_time']:.1f}s - {metadata['end_time']:.1f}s)")
        else:
            print("  No results found")

def demo_advanced_search(indexer: ShotVectorIndexer):
    """Demonstrate advanced search features."""
    print("=== Advanced Search Demo ===")
    
    # Example 1: Search with duration filter
    print("\n1. Searching for 'family' scenes with duration > 5 seconds:")
    results = indexer.search_shots("family", n_results=5, min_duration=5.0)
    for i, result in enumerate(results, 1):
        metadata = result['metadata']
        print(f"   {i}. Shot {metadata['shot_number']} - Duration: {metadata['duration']:.1f}s")
    
    # Example 2: Search with video filter
    print("\n2. Searching for 'happy' scenes in specific video:")
    results = indexer.search_shots("happy", n_results=3, video_name="Hair Love")
    for i, result in enumerate(results, 1):
        metadata = result['metadata']
        print(f"   {i}. Shot {metadata['shot_number']} from '{metadata['video_name']}'")
    
    # Example 3: Search with shot number filter
    print("\n3. Searching for 'bathroom' scenes in early shots (shot number < 20):")
    results = indexer.search_shots("bathroom", n_results=3)
    filtered_results = [r for r in results if r['metadata']['shot_number'] < 20]
    for i, result in enumerate(filtered_results, 1):
        metadata = result['metadata']
        print(f"   {i}. Shot {metadata['shot_number']} from '{metadata['video_name']}'")

def demo_query_examples(indexer: ShotVectorIndexer):
    """Show various query examples."""
    print("=== Query Examples Demo ===")
    
    query_examples = [
        "happy family scene",
        "sad emotional moment",
        "bathroom hair styling",
        "young girl with curly hair",
        "cat in the scene",
        "bedroom setting",
        "hospital scene",
        "father and daughter",
        "pink dress",
        "tablet or laptop",
        "mirror reflection",
        "window view",
        "nighttime scene",
        "daylight scene",
        "close-up shot",
        "wide angle shot",
        "action sequence",
        "calm peaceful moment"
    ]
    
    print("Available query examples:")
    for i, query in enumerate(query_examples, 1):
        print(f"  {i:2d}. '{query}'")
    
    print("\nTry any of these queries with the --query parameter!")

def main():
    # parser = argparse.ArgumentParser(description="Demo script for Shot Vector Indexer")
    # parser.add_argument("--action", choices=["index", "search", "demo", "examples"], required=True,
    #                    help="Action to perform")
    # parser.add_argument("--input", type=Path, help="Input CSV file or directory for indexing")
    # parser.add_argument("--query", type=str, help="Search query")
    # parser.add_argument("--results", type=int, default=5, help="Number of search results")
    # parser.add_argument("--db-type", default="chroma", choices=["chroma", "pinecone"],
    #                    help="Vector database type")
    # parser.add_argument("--model", default="all-MiniLM-L6-v2", 
    #                    help="Embedding model name")
    
    # args = parser.parse_args()
    db_type = 'chroma'
    model = 'all-MiniLM-L6-v2'
    action = 'demo'
    # Use absolute path to the workspace root
    workspace_root = Path(__file__).parent.parent
    input = workspace_root / 'processed_videos_output_module2_2_scenes'

    
    # Initialize indexer
    try:
        indexer = ShotVectorIndexer(
            db_type=db_type,
            embedding_model=model
        )
        print(f"✅ Initialized indexer with {db_type} database and {model} model")
    except Exception as e:
        print(f"❌ Failed to initialize indexer: {e}")
        return
    
    # Perform requested action
    if action == "index":
        if not input:
            print("❌ --input required for indexing action")
            return
        demo_indexing(indexer, input)
        
    elif action == "search":
        # if not query:
        #     print("❌ --query required for search action")
        #     return
        # demo_searching(indexer, query, results)
        pass
        
    elif action == "demo":
        if not input:
            print("❌ --input required for demo action")
            return
        
        # Run full demo
        print("🚀 Running Full Demo")
        print("=" * 50)
        
        # Step 1: Index data
        demo_indexing(indexer, input)
        print("\n" + "=" * 50)
        
        # Step 2: Basic search
        demo_searching(indexer, "Two ladies with black attire sitting in church", 3)
        print("\n" + "=" * 50)
        
        demo_searching(indexer, "Girl walking in cemetrey at sun rise", 3)
        print("\n" + "=" * 50)

        

        demo_searching(indexer, "scene where hair love is shown in youtube channel", 3)
        print("\n" + "=" * 50)

        demo_searching(indexer, "scene where man putting a red hat and girl is surprised", 3)
        print("\n" + "=" * 50)

        demo_searching(indexer, "a closeup scene where man is combing hair ", 3)
        print("\n" + "=" * 50)
        
        # Step 3: Metadata search
        demo_metadata_search(indexer)
        print("\n" + "=" * 50)
        
        # Step 4: Advanced search
        demo_advanced_search(indexer)
        print("\n" + "=" * 50)
        
        # Step 5: Query examples
        demo_query_examples(indexer)
        
    elif action == "examples":
        demo_query_examples(indexer)

if __name__ == "__main__":
    main() 