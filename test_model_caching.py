#!/usr/bin/env python3
"""
Test script to demonstrate model caching functionality.
This script shows how the model is only downloaded once and reused from cache.
"""

import time
from pathlib import Path
from src.videoqa.shot_vector_indexer import ShotVectorIndexer, clear_model_cache, get_cached_models

def test_model_caching():
    """Test that models are cached and not re-downloaded."""
    
    print("=== Model Caching Test ===\n")
    
    # Clear any existing cache
    clear_model_cache()
    print(f"Initial cache: {get_cached_models()}")
    
    # Create first indexer - this should download the model
    print("\n1. Creating first ShotVectorIndexer...")
    start_time = time.time()
    indexer1 = ShotVectorIndexer(
        db_type="chroma",
        embedding_model="all-MiniLM-L6-v2"
    )
    time1 = time.time() - start_time
    print(f"   Time taken: {time1:.2f} seconds")
    print(f"   Cache after first load: {get_cached_models()}")
    
    # Create second indexer - this should use cached model
    print("\n2. Creating second ShotVectorIndexer with same model...")
    start_time = time.time()
    indexer2 = ShotVectorIndexer(
        db_type="chroma",
        embedding_model="all-MiniLM-L6-v2"
    )
    time2 = time.time() - start_time
    print(f"   Time taken: {time2:.2f} seconds")
    print(f"   Cache after second load: {get_cached_models()}")
    
    # Create third indexer with different model - this should download new model
    print("\n3. Creating third ShotVectorIndexer with different model...")
    start_time = time.time()
    indexer3 = ShotVectorIndexer(
        db_type="chroma",
        embedding_model="all-mpnet-base-v2"
    )
    time3 = time.time() - start_time
    print(f"   Time taken: {time3:.2f} seconds")
    print(f"   Cache after third load: {get_cached_models()}")
    
    # Test statistics
    print("\n4. Getting statistics from indexers...")
    stats1 = indexer1.get_statistics()
    print(f"   Indexer1 stats: {stats1}")
    
    # Test cache management
    print("\n5. Testing cache management...")
    print(f"   Cached models before clearing: {get_cached_models()}")
    ShotVectorIndexer.clear_model_cache()
    print(f"   Cached models after clearing: {get_cached_models()}")
    
    print("\n=== Test Complete ===")
    print(f"First load time: {time1:.2f}s")
    print(f"Second load time: {time2:.2f}s")
    print(f"Third load time: {time3:.2f}s")
    print(f"Cache speedup: {time1/time2:.1f}x faster for second load")

if __name__ == "__main__":
    test_model_caching() 