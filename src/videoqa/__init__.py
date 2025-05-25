"""VideoQA package for video ingestion and preprocessing."""

import os
import sys

# Set tokenizers parallelism before importing any HuggingFace libraries
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Ensure ffmpeg is in the PATH
os.environ["PATH"] = "/opt/homebrew/bin:" + os.environ.get("PATH", "")

# Now import ffmpeg-related modules
from .video_processor import *  # noqa
from .main import *  # noqa

__version__ = "0.1.0" 