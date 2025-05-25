# video_analysis_project/src/video_ingestion/utils.py

import shutil
import logging

# Configure basic logging for utilities
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')


def check_ffmpeg_availability():
    """
    Checks if the ffmpeg and ffprobe commands are available on the system PATH.
    Raises FileNotFoundError if ffmpeg or ffprobe is not found.
    """
    ffmpeg_path = shutil.which("ffmpeg")
    ffprobe_path = shutil.which("ffprobe")

    if ffmpeg_path is None:
        error_msg = (
            "FFmpeg not found. Please install FFmpeg and ensure it is in your system's PATH. "
            "Download from https://ffmpeg.org/download.html"
        )
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    if ffprobe_path is None:
        error_msg = (
            "ffprobe not found (usually part of FFmpeg). Please ensure it is in your system's PATH."
        )
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
        
    logger.info(f"FFmpeg found at: {ffmpeg_path}")
    logger.info(f"ffprobe found at: {ffprobe_path}")
    return True

# You can add other utility functions here, e.g., for creating directory structures robustly.