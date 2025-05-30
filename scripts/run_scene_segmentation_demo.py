# video_analysis_project/scripts/run_scene_segmentation_demo.py

import sys
import os
from pathlib import Path
import json
import logging

# Add src to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

try:
    # Import from both modules
    from video_ingestion import process_video_pipeline as run_ingestion
    from video_ingestion.utils import check_ffmpeg_availability as check_ffmpeg_ingestion
    from scene_segmentation import segment_video_into_scenes
    from scene_segmentation.feature_extractor import DEVICE as SCENE_SEG_DEVICE # To log device
except ImportError as e:
    print(f"Error importing packages: {e}")
    print(f"Ensure 'uv pip install -e .' has been run from '{project_root}'.")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)


# (Re-use the dummy video creation function from run_ingestion_demo.py if needed,
# or assume the video exists)
def create_dummy_video_if_not_exists(video_filepath: str, duration: int = 10) -> bool:
    video_p = Path(video_filepath)
    if video_p.exists():
        logger.info(f"Video file '{video_filepath}' already exists.")
        return True
    logger.info(f"Attempting to create dummy video: {video_filepath}")
    try:
        check_ffmpeg_ingestion()
        video_p.parent.mkdir(parents=True, exist_ok=True)
        import subprocess
        cmd = [
            "ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c=red:s=320x240:d={duration//2}",
            "-f", "lavfi", "-i", f"color=c=green:s=320x240:d={duration//2}",
            "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]",
            "-map", "[v]",
            "-f", "lavfi", "-i", f"anullsrc=channel_layout=stereo:sample_rate=44100:d={duration}",
            "-vcodec", "libx264", "-pix_fmt", "yuv420p",
            "-acodec", "aac", "-ar", "44100", "-shortest", str(video_p)
        ] # Creates a 2-part video (red then green) to ensure some scene changes
        process = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if process.returncode == 0:
            logger.info(f"Successfully created dummy video: {video_filepath}")
            return True
        else:
            logger.error(f"Failed to create dummy video. FFmpeg stderr: {process.stderr}")
            return False
    except Exception as e:
        logger.error(f"Error creating dummy video '{video_filepath}': {e}", exc_info=True)
        return False

if __name__ == "__main__":
    logger.info("Starting Scene Segmentation Demo Script...")
    logger.info(f"Scene Segmentation models will run on: {SCENE_SEG_DEVICE}")

    # --- Configuration ---
    demo_video_dir = project_root / "sample_videos"
    sample_video_filename = "Hair Love.webm" # Use a different name for this demo
    original_video_path = str(demo_video_dir / sample_video_filename)

    # Output directory for Module 2.1 (Video Ingestion)
    ingestion_base_output_dir = str(project_root / "processed_videos_output")
    
    # Output directory for Module 2.2 (Scene Segmentation)
    scene_seg_base_output_dir = str(project_root / "processed_videos_output_module2_2_scenes")

    # 1. Ensure dummy video exists
    if not create_dummy_video_if_not_exists(original_video_path, duration=20): # Longer video for more shots
        logger.critical("Could not create or find sample video. Demo aborted.")
        sys.exit(1)

    # 2. Run Ingestion Pipeline (Module 2.1) first if its output is needed and not present
    video_name_stem = Path(original_video_path).stem
    ingestion_results_dir_for_video = Path(ingestion_base_output_dir) / video_name_stem
    ingestion_summary_file = ingestion_results_dir_for_video / f"{video_name_stem}_processing_summary.json"

    if not ingestion_summary_file.exists():
        logger.info(f"Ingestion results not found for {video_name_stem}. Running Ingestion Pipeline first...")
        ingestion_results = run_ingestion(
            video_path=original_video_path,
            base_output_dir=ingestion_base_output_dir,
            frame_rate=5, # Higher frame rate might be useful for shot boundary later
            whisper_model="tiny",
            force_reprocessing=True # Force for demo consistency
        )
        if ingestion_results.get("status") not in ["completed_successfully", "completed_with_errors", "skipped_existing"]:
            logger.critical(f"Video ingestion failed: {ingestion_results.get('error', 'Unknown ingestion error')}. Cannot proceed.")
            sys.exit(1)
        elif ingestion_results.get("status") == "completed_with_errors" and not (ingestion_results_dir_for_video/"audio").exists():
            logger.critical(f"Video ingestion completed with errors, and essential outputs might be missing. Cannot proceed.")
            sys.exit(1)

    else:
        logger.info(f"Found existing ingestion results for {video_name_stem} at {ingestion_results_dir_for_video}")

    # 3. Run Scene Segmentation Pipeline (Module 2.2)
    logger.info("\n--- Starting Scene Segmentation Pipeline ---")
    try:
        scene_results = segment_video_into_scenes(
            video_path=original_video_path,
            ingestion_output_dir=ingestion_base_output_dir, # Pass the base dir
            scene_segmentation_output_dir=scene_seg_base_output_dir,
            shot_detector_threshold=25.0, # Lower threshold = more sensitive shot detection
            min_shot_len_frames=10,
            num_keyframes_per_shot=10,     # 1 keyframe (middle) for visual features per shot
            modality_weights={"visual": 1.0, "audio": 0.0, "textual": 0.0}, # Adjust as needed
            scene_similarity_threshold=0.15, # If combined similarity is below this, new scene
            min_shots_per_scene=1,         # Allow single-shot scenes
            shotdetection_reprocessing=False,
            feature_extraction_reprocessing=True
        )
        
        logger.info("\n--- Final Scene Segmentation Results ---")
        # logger.info(json.dumps(scene_results, indent=2))

        if scene_results.get("status") == "pipeline_completed_successfully":
            logger.info("Scene segmentation demo completed successfully!")
        else:
            logger.warning("Scene segmentation demo completed with issues or errors. Check logs.")

    except Exception as e:
        logger.critical(f"An error occurred during the scene segmentation demo: {e}", exc_info=True)