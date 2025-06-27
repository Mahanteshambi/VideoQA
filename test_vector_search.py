#!/usr/bin/env python3
"""
Simple test script for the Shot Vector Indexer

This script tests the basic functionality of the vector search system
using the existing CSV data from the Hair Love video.
"""

import sys
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent / "src"))

def test_basic_functionality():
    """Test basic indexing and search functionality."""
    print("🧪 Testing Shot Vector Indexer Basic Functionality")
    print("=" * 60)
    
    try:
        from videoqa.shot_vector_indexer import ShotVectorIndexer, MetadataParser
        
        # Initialize indexer
        print("1. Initializing indexer...")
        indexer = ShotVectorIndexer(
            db_type="chroma",
            embedding_model="all-MiniLM-L6-v2"
        )
        print("✅ Indexer initialized successfully")
        
        # Test CSV parsing
        print("\n2. Testing CSV parsing...")
        csv_path = Path("processed_videos_output_module2_2_scenes/Hair Love/Hair Love_shot_metadata.csv")
        
        if not csv_path.exists():
            print(f"❌ CSV file not found: {csv_path}")
            return False
        
        shots = MetadataParser.parse_csv_to_metadata(csv_path, "Hair Love")
        print(f"✅ Parsed {len(shots)} shots from CSV")
        
        if len(shots) == 0:
            print("❌ No shots parsed from CSV")
            return False
        
        # Test text generation
        print("\n3. Testing text generation...")
        sample_shot = shots[0]
        print(f"Sample shot {sample_shot.shot_number}:")
        print(f"  Description: {sample_shot.shot_description}")
        print(f"  Combined text length: {len(sample_shot.combined_text)} characters")
        print(f"  Hash: {sample_shot.metadata_hash}")
        print("✅ Text generation working")
        
        # Test indexing
        print("\n4. Testing indexing...")
        success = indexer.index_csv_file(csv_path, "Hair Love")
        if success:
            print("✅ Indexing successful")
        else:
            print("❌ Indexing failed")
            return False
        
        # Test searching
        print("\n5. Testing search functionality...")
        test_queries = [
            "happy family scene",
            "bathroom hair styling",
            "young girl with curly hair",
            "cat in the scene"
        ]
        
        for query in test_queries:
            print(f"\n   Searching for: '{query}'")
            results = indexer.search_shots(query, n_results=3)
            if results:
                print(f"   ✅ Found {len(results)} results")
                for i, result in enumerate(results, 1):
                    metadata = result['metadata']
                    print(f"     {i}. Shot {metadata['shot_number']} (score: {result['distance']:.4f})")
            else:
                print(f"   ⚠️  No results for '{query}'")
        
        print("\n✅ All basic tests passed!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please install required dependencies: pip install -r requirements_vector_search.txt")
        return False
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_metadata_search():
    """Test metadata-based search functionality."""
    print("\n🧪 Testing Metadata-Based Search")
    print("=" * 60)
    
    try:
        from videoqa.shot_vector_indexer import ShotVectorIndexer
        
        indexer = ShotVectorIndexer(db_type="chroma", embedding_model="all-MiniLM-L6-v2")
        
        # Test metadata search
        print("1. Testing metadata search...")
        search_tests = [
            {"name": "Family scenes", "params": {"genre": "Family"}},
            {"name": "Happy mood", "params": {"mood": "happy"}},
            {"name": "Home setting", "params": {"setting": "home"}},
            {"name": "Hair care content", "params": {"content": "hair care"}}
        ]
        
        for test in search_tests:
            print(f"\n   Testing: {test['name']}")
            results = indexer.search_by_metadata(**test['params'], n_results=2)
            if results:
                print(f"   ✅ Found {len(results)} results")
                for i, result in enumerate(results, 1):
                    metadata = result['metadata']
                    print(f"     {i}. Shot {metadata['shot_number']} from {metadata['video_name']}")
            else:
                print(f"   ⚠️  No results for {test['name']}")
        
        print("\n✅ Metadata search tests completed!")
        return True
        
    except Exception as e:
        print(f"❌ Metadata search test failed: {e}")
        return False

def test_advanced_features():
    """Test advanced search features."""
    print("\n🧪 Testing Advanced Features")
    print("=" * 60)
    
    try:
        from videoqa.shot_vector_indexer import ShotVectorIndexer
        
        indexer = ShotVectorIndexer(db_type="chroma", embedding_model="all-MiniLM-L6-v2")
        
        # Test filtering
        print("1. Testing search with filters...")
        
        # Duration filter
        print("   Testing duration filter...")
        results = indexer.search_shots("family", n_results=5, min_duration=5.0)
        if results:
            print(f"   ✅ Found {len(results)} shots with duration > 5s")
            for result in results:
                metadata = result['metadata']
                print(f"     Shot {metadata['shot_number']}: {metadata['duration']:.1f}s")
        
        # Video name filter
        print("\n   Testing video name filter...")
        results = indexer.search_shots("happy", n_results=3, video_name="Hair Love")
        if results:
            print(f"   ✅ Found {len(results)} shots from Hair Love")
            for result in results:
                metadata = result['metadata']
                print(f"     Shot {metadata['shot_number']} from {metadata['video_name']}")
        
        print("\n✅ Advanced feature tests completed!")
        return True
        
    except Exception as e:
        print(f"❌ Advanced feature test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Starting Shot Vector Indexer Tests")
    print("=" * 60)
    
    # Run tests
    test1_passed = test_basic_functionality()
    test2_passed = test_metadata_search()
    test3_passed = test_advanced_features()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)
    print(f"Basic Functionality: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"Metadata Search: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    print(f"Advanced Features: {'✅ PASSED' if test3_passed else '❌ FAILED'}")
    
    all_passed = test1_passed and test2_passed and test3_passed
    
    if all_passed:
        print("\n🎉 All tests passed! The vector search system is working correctly.")
        print("\nNext steps:")
        print("1. Try the demo script: python scripts/demo_shot_vector_search.py --action demo --input processed_videos_output_module2_2_scenes/")
        print("2. Experiment with different queries")
        print("3. Check the VECTOR_SEARCH_README.md for more information")
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")
        print("\nTroubleshooting:")
        print("1. Install dependencies: pip install -r requirements_vector_search.txt")
        print("2. Ensure the CSV file exists: processed_videos_output_module2_2_scenes/Hair Love/Hair Love_shot_metadata.csv")
        print("3. Check that ChromaDB is working correctly")

if __name__ == "__main__":
    main() 