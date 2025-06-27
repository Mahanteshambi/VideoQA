#!/usr/bin/env python3
"""
Test script for the updated scene segmentation pipeline with metadata extraction and CSV output.
"""

import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from scene_segmentation.pipeline import segment_video_into_scenes

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_pipeline():
    """Test the updated pipeline with metadata extraction."""
    
    # Test parameters
    video_path = "sample_videos/Hair Love.mp4"
    ingestion_output_dir = "processed_videos_output"
    scene_segmentation_output_dir = "processed_videos_output_module2_2_scenes"
    
    # Check if video exists
    if not Path(video_path).exists():
        print(f"Error: Video file not found at {video_path}")
        return
    
    print("Starting pipeline test with metadata extraction...")
    print(f"Video: {video_path}")
    print(f"Ingestion output dir: {ingestion_output_dir}")
    print(f"Scene segmentation output dir: {scene_segmentation_output_dir}")
    
    try:
        # Run the pipeline
        results = segment_video_into_scenes(
            video_path=video_path,
            ingestion_output_dir=ingestion_output_dir,
            scene_segmentation_output_dir=scene_segmentation_output_dir,
            shot_detector_threshold=30.0,
            min_shot_len_frames=15,
            num_keyframes_per_shot=1,
            modality_weights={"visual": 0.9, "audio": 0.0, "textual": 0.1},
            scene_similarity_threshold=0.4,
            min_shots_per_scene=2,
            shotdetection_reprocessing=False,  # Use cached if available
            feature_extraction_reprocessing=True,  # Use cached if available
            vllm_annotator_type="smolvlm"
        )
        
        print("\n=== Pipeline Results ===")
        print(f"Status: {results['status']}")
        print(f"Shot detection: {results['shot_detection']['status']} ({results['shot_detection']['count']} shots)")
        print(f"Feature extraction: {results['feature_extraction']['status']} ({results['feature_extraction']['shots_processed']} processed)")
        print(f"Scene grouping: {results['scene_grouping']['status']} ({results['scene_grouping']['scene_count']} scenes)")
        
        if results['errors']:
            print(f"\nErrors: {len(results['errors'])}")
            for error in results['errors']:
                print(f"  - {error}")
        
        # Check for CSV output
        video_name = Path(video_path).stem
        csv_file = Path(scene_segmentation_output_dir) / video_name / f"{video_name}_shot_metadata.csv"
        
        if csv_file.exists():
            print(f"\n✅ CSV metadata file created: {csv_file}")
            
            # Show first few lines of CSV
            print("\n=== First few lines of CSV ===")
            with open(csv_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for i, line in enumerate(lines[:5]):  # Show first 5 lines
                    print(f"Line {i+1}: {line.strip()}")
        else:
            print(f"\n❌ CSV metadata file not found: {csv_file}")
        
    except Exception as e:
        print(f"Error running pipeline: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pipeline() 