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
    shot1_features: dict, 
    shot2_features: dict, 
    weights: dict = {"visual": 0.5, "audio": 0.25, "textual": 0.25}
) -> float:
    """
    Calculates the weighted multimodal similarity between two shots.

    Args:
        shot1_features (dict): Multimodal features for shot 1 (e.g., {"visual": ndarray, "audio": ndarray, ...}).
        shot2_features (dict): Multimodal features for shot 2.
        weights (dict): Weights for each modality (e.g., {"visual": 0.5, "audio": 0.25, "textual": 0.25}).
                        Weights should sum to 1 for a normalized score between 0 and 1.

    Returns:
        float: The combined multimodal similarity score (between 0 and 1).
    """
    if not shot1_features or not shot2_features:
        logger.warning("One or both shot feature sets are missing. Returning 0 similarity.")
        return 0.0

    total_similarity = 0.0
    total_weight_applied = 0.0 # To normalize if some features are missing

    # Visual Similarity
    if "visual" in weights and weights["visual"] > 0:
        sim_v = calculate_cosine_similarity(shot1_features.get("visual"), shot2_features.get("visual"))
        total_similarity += weights["visual"] * sim_v
        total_weight_applied += weights["visual"]
        logger.debug(f"Sim_V: {sim_v:.3f}")

    # Audio Similarity
    if "audio" in weights and weights["audio"] > 0:
        # For placeholder audio (single RMS value), cosine similarity might not be ideal.
        # Let's use a scaled difference for now if vectors are 1D.
        # This part needs refinement if using proper audio embeddings.
        audio1 = shot1_features.get("audio")
        audio2 = shot2_features.get("audio")
        sim_a = 0.0
        if audio1 is not None and audio2 is not None:
            if audio1.ndim == 1 and audio1.size == 1 and audio2.ndim == 1 and audio2.size == 1: # Placeholder RMS
                # Simple normalized difference: 1 - abs(v1-v2) / (max_possible_diff_or_scale)
                # Assuming RMS is normalized roughly between 0 and 1 (or some small positive range)
                diff = np.abs(audio1[0] - audio2[0])
                sim_a = max(0.0, 1.0 - diff) # Crude similarity for scalar feature
            else: # Assume it's a proper embedding
                sim_a = calculate_cosine_similarity(audio1, audio2)
        
        total_similarity += weights["audio"] * sim_a
        total_weight_applied += weights["audio"]
        logger.debug(f"Sim_A: {sim_a:.3f} (using {'placeholder' if audio1 is not None and audio1.size==1 else 'embedding'} logic)")


    # Textual Similarity
    if "textual" in weights and weights["textual"] > 0:
        sim_t = calculate_cosine_similarity(shot1_features.get("textual"), shot2_features.get("textual"))
        total_similarity += weights["textual"] * sim_t
        total_weight_applied += weights["textual"]
        logger.debug(f"Sim_T: {sim_t:.3f}")

    if total_weight_applied == 0: # No valid modalities or weights to compare
        return 0.0
        
    final_score = total_similarity / total_weight_applied # Normalize by sum of weights used
    logger.debug(f"Shot {shot1_features.get('shot_number','S1')} vs Shot {shot2_features.get('shot_number','S2')} - TotalSim: {final_score:.3f}")
    return final_score