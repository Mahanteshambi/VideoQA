# video_analysis_project/src/video_ingestion/__init__.py

# Expose the main pipeline function for easier access
from .pipeline import process_video_pipeline
from .utils import check_ffmpeg_availability

# You can also define a __all__ variable to specify what gets imported with 'from video_ingestion import *'
__all__ = ['process_video_pipeline', 'check_ffmpeg_availability']

print("Video Ingestion Package Initialized")