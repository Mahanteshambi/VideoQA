"""VideoQA package for video ingestion and preprocessing."""

import os
import sys

# Add ffmpeg binary path to environment
os.environ["PATH"] = "/opt/homebrew/bin:" + os.environ.get("PATH", "")

# Add ffmpeg binary path to system path
if "/opt/homebrew/bin" not in sys.path:
    sys.path.append("/opt/homebrew/bin")

from .video_processor import *  # noqa
from .main import *  # noqa

__version__ = "0.1.0" 