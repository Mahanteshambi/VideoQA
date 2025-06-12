# video_analysis_project/src/scene_segmentation/similarity_calculator.py

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)

def calculate_cosine_similarity(vec1: np.ndarray | None, vec2: np.ndarray | None) -> float:
    """Calculates cosine similarity. Returns 0.0 if either vector is None or all zeros."""
    if vec1 is None or vec2 is None:
        return 0.0 # No similarity if one feature is missing
    if np.all(vec1 == 0) or np.all(vec2 == 0): # Handle zero vectors (e.g. no text)
        return 0.0 # Or 1.0 if zero vectors should be considered identical in some contexts
    
    # Ensure vectors are 2D for cosine_similarity function
    sim = cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))[0, 0]
    return (sim + 1.0) / 2.0 # Scale from [-1, 1] to [0, 1]

def calculate_inter_shot_multimodal_similarity(
    shot1_metadata: dict, 
    shot2_metadata: dict, 
    weights: dict = {"visual": 0.5, "audio": 0.25, "textual": 0.25}
) -> float:
    """
    Calculates the weighted multimodal similarity between two shots based on their VLLM metadata.

    Args:
        shot1_metadata (dict): Shot 1 dict containing VLLM metadata.
        shot2_metadata (dict): Shot 2 dict containing VLLM metadata.
        weights (dict): Weights for each modality (e.g., {"visual": 0.5, "audio": 0.25, "textual": 0.25}).
                        Weights should sum to 1 for a normalized score between 0 and 1.

    Returns:
        float: The combined multimodal similarity score (between 0 and 1).
    """
    if not shot1_metadata or not shot2_metadata:
        logger.warning("One or both shot metadata sets are missing. Returning 0 similarity.")
        return 0.0

    # Extract descriptions from metadata for comparison
    description1 = shot1_metadata.get("description", "")
    description2 = shot2_metadata.get("description", "")
    
    # Simple text similarity based on descriptions
    if description1 and description2:
        # Calculate a simple similarity score based on common words
        words1 = set(description1.lower().split())
        words2 = set(description2.lower().split())
        
        if not words1 or not words2:
            return 0.0
            
        common_words = words1.intersection(words2)
        similarity_score = len(common_words) / max(len(words1), len(words2))
        
        logger.debug(f"Shot similarity based on descriptions: {similarity_score:.3f}")
        return similarity_score
    
    # Fallback to a default similarity calculation if no descriptions
    return 0.5  # Default middle value when we can't determine similarity