"""Scene Segmentation package for video analysis."""

import os
import logging

# Set tokenizers parallelism before importing any HuggingFace libraries
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)

from .feature_extractor import *  # noqa
from .shot_detector import *      # noqa
from .pipeline import *          # noqa

print("Scene Segmentation Package Initialized")
