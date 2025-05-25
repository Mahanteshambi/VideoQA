# video_analysis_project/scripts/run_ingestion_demo.py

import sys
import os
from pathlib import Path
import json
import logging

# Add the src directory to Python's path so it can find the video_ingestion package
# This is a common way to handle imports when running scripts from outside the package root
# A better way for larger projects might be to install the package in editable mode (uv pip install -e .)
# and then imports work naturally.
# For this script, we'll do a simple path modification.
project_root = Path(__file__).resolve().parent.parent # Should be video_analysis_project/
sys.path.insert(0, str(project_root / "src"))

try:
    from video_ingestion import process_video_pipeline, check_ffmpeg_availability
except ImportError as e:
    print(f"Error importing video_ingestion package: {e}")
    print(f"Ensure your PYTHONPATH is set correctly or run 'uv pip install -e .' from the project root '{project_root}'.")
    sys.exit(1)

# Configure basic logging for the demo script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)


def create_dummy_video_if_not_exists(video_filepath: str, duration: int = 2) -> bool:
    """Creates a short dummy MP4 video if it doesn't exist."""
    video_p = Path(video_filepath)
    if video_p.exists():
        logger.info(f"Video file '{video_filepath}' already exists.")
        return True
    
    logger.info(f"Attempting to create dummy video: {video_filepath}")
    try:
        check_ffmpeg_availability() # This will raise if not found
        
        # Ensure parent directory exists for the dummy video
        video_p.parent.mkdir(parents=True, exist_ok=True)

        # Use ffmpeg directly via os.system or subprocess for simplicity in dummy creation
        # Using ffmpeg-python for this basic task can be verbose.
        # For more complex ffmpeg commands, ffmpeg-python is great.
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output files without asking
            "-f", "lavfi", "-i", f"color=c=blue:s=320x240:d={duration}", # Blue video
            "-f", "lavfi", "-i", f"anullsrc=channel_layout=stereo:sample_rate=44100:d={duration}", # Silent audio
            "-vcodec", "libx264", "-pix_fmt", "yuv420p",
            "-acodec", "aac", "-ar", "44100",
            "-shortest", # Finish encoding when the shortest input stream ends
            str(video_p)
        ]
        logger.info(f"Executing FFmpeg command: {' '.join(cmd)}")
        
        import subprocess
        process = subprocess.run(cmd, capture_output=True, text=True, check=False)

        if process.returncode == 0:
            logger.info(f"Successfully created dummy video: {video_filepath}")
            return True
        else:
            logger.error(f"Failed to create dummy video. FFmpeg stderr: {process.stderr}")
            return False
    except FileNotFoundError:
        logger.error("FFmpeg not found. Cannot create dummy video.")
        return False
    except Exception as e:
        logger.error(f"Error creating dummy video '{video_filepath}': {e}", exc_info=True)
        return False

if __name__ == "__main__":
    logger.info("Starting Video Ingestion Demo Script...")

    # --- Configuration for Demo ---
    # Define paths relative to this script or use absolute paths
    demo_video_dir = project_root / "sample_videos"
    sample_video_filename = "Hair Love.webm"
    sample_video_path = demo_video_dir / sample_video_filename
    
    base_output_directory = project_root / "processed_videos_output"

    # Attempt to create the dummy video
    if not create_dummy_video_if_not_exists(str(sample_video_path), duration=5):
        logger.critical(f"Could not create or find sample video at {sample_video_path}. Demo aborted.")
        sys.exit(1)
    
    logger.info(f"Using sample video: {sample_video_path}")
    logger.info(f"Output will be stored in: {base_output_directory}")

    # --- Run the processing pipeline ---
    try:
        processing_results = process_video_pipeline(
            video_path=str(sample_video_path),
            base_output_dir=str(base_output_directory),
            frame_rate=1,                 # Extract 1 frame per second
            whisper_model="tiny",         # Use "tiny" for faster demo
            # skip_frames=False,
            # skip_audio_extraction=False,
            # skip_transcription=False,
            # skip_metadata=False,
            force_reprocessing=True      # Set to False to skip if output dir exists
        )
        logger.info("\n--- Final Processing Results from Pipeline ---")
        # Pretty print the JSON results
        logger.info(json.dumps(processing_results, indent=2, ensure_ascii=False))

        if processing_results.get("status") == "completed_successfully":
            logger.info("Demo completed successfully!")
        elif processing_results.get("status") == "completed_with_errors":
             logger.warning("Demo completed with some errors. Check logs and results summary.")
        elif processing_results.get("status") == "skipped_existing":
            logger.info("Demo skipped processing as output already exists. Use force_reprocessing=True in the script to override.")
        else:
            logger.error(f"Demo encountered an issue: {processing_results.get('error', 'Unknown error')}")


    except Exception as e:
        logger.critical(f"An error occurred during the demo run: {e}", exc_info=True)