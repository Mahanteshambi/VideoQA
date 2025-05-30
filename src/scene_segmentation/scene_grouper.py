# video_analysis_project/src/scene_segmentation/scene_grouper.py

import logging
from .similarity_calculator import calculate_inter_shot_multimodal_similarity

logger = logging.getLogger(__name__)

def group_shots_into_scenes(
    shots_with_features: list[dict], # List of shots, each dict containing its features
    similarity_threshold: float,
    modality_weights: dict,
    min_shots_per_scene: int = 1 # Minimum number of shots to form a scene
) -> list[dict]:
    """
    Groups a list of shots (with their multimodal features) into scenes based on
    inter-shot similarity.

    Args:
        shots_with_features (list[dict]): List of shot dictionaries. Each shot dict must
                                         contain its multimodal features under a 'features' key,
                                         and also original shot info like 'start_time_seconds',
                                         'end_time_seconds', 'shot_number'.
        similarity_threshold (float): If similarity between last shot of current scene and
                                      next shot falls below this, a new scene starts. (Range 0-1)
        modality_weights (dict): Weights for combining multimodal similarities.
        min_shots_per_scene (int): Minimum shots required to form a distinct scene.
                                   Single shots can be scenes if this is 1.

    Returns:
        list[dict]: A list of scenes. Each scene is a dictionary containing:
                    'scene_number', 'start_time_seconds', 'end_time_seconds',
                    'duration_seconds', 'num_shots', 'shots' (list of original shot dicts).
    """
    if not shots_with_features:
        logger.warning("No shots provided to group into scenes.")
        return []

    scenes = []
    current_scene_shots = []
    scene_counter = 1

    # Always start the first scene with the first shot
    current_scene_shots.append(shots_with_features[0])

    for i in range(len(shots_with_features) - 1):
        current_shot = shots_with_features[i]
        next_shot = shots_with_features[i+1]

        # Calculate similarity between the last shot of the *current forming scene* and the next shot
        # For simplicity here, we compare current_shot (i) with next_shot (i+1)
        # A more advanced approach might compare next_shot with an aggregated feature of current_scene_shots
        
        similarity = calculate_inter_shot_multimodal_similarity(
            current_shot['features'], 
            next_shot['features'], 
            modality_weights
        )

        if similarity < similarity_threshold:
            # End current scene if it meets min_shots criteria
            if len(current_scene_shots) >= min_shots_per_scene:
                scene_start_shot = current_scene_shots[0]
                scene_end_shot = current_scene_shots[-1]
                scenes.append({
                    "scene_number": scene_counter,
                    "start_time_seconds": scene_start_shot["start_time_seconds"],
                    "end_time_seconds": scene_end_shot["end_time_seconds"],
                    "duration_seconds": scene_end_shot["end_time_seconds"] - scene_start_shot["start_time_seconds"],
                    "num_shots": len(current_scene_shots),
                    "shots": list(current_scene_shots) # Store a copy
                })
                scene_counter += 1
            else:
                # If current scene is too short, merge it with the next one by not breaking here
                # Or, handle it as a very short scene if that's desired.
                # For now, we effectively extend the scene if it's too short.
                # This logic might need refinement based on desired behavior for very short "scenes".
                logger.debug(f"Potential scene break after shot {current_shot['shot_number']} (sim: {similarity:.3f}), but current scene too short ({len(current_scene_shots)} shots). Extending.")

            current_scene_shots = [] # Start a new scene buffer
        
        current_scene_shots.append(next_shot)

    # Add the last formed scene
    if current_scene_shots and len(current_scene_shots) >= min_shots_per_scene:
        scene_start_shot = current_scene_shots[0]
        scene_end_shot = current_scene_shots[-1]
        scenes.append({
            "scene_number": scene_counter,
            "start_time_seconds": scene_start_shot["start_time_seconds"],
            "end_time_seconds": scene_end_shot["end_time_seconds"],
            "duration_seconds": scene_end_shot["end_time_seconds"] - scene_start_shot["start_time_seconds"],
            "num_shots": len(current_scene_shots),
            "shots": list(current_scene_shots)
        })
    elif current_scene_shots and scenes: # Append remaining short segment to the last valid scene
        logger.info(f"Appending {len(current_scene_shots)} remaining short shots to the last scene ({scenes[-1]['scene_number']}).")
        scenes[-1]["shots"].extend(current_scene_shots)
        scenes[-1]["end_time_seconds"] = current_scene_shots[-1]["end_time_seconds"]
        scenes[-1]["duration_seconds"] = scenes[-1]["end_time_seconds"] - scenes[-1]["start_time_seconds"]
        scenes[-1]["num_shots"] = len(scenes[-1]["shots"])

    elif current_scene_shots and not scenes: # Only one scene, shorter than min_shots_per_scene but it's all we have
         scene_start_shot = current_scene_shots[0]
         scene_end_shot = current_scene_shots[-1]
         scenes.append({
            "scene_number": scene_counter,
            "start_time_seconds": scene_start_shot["start_time_seconds"],
            "end_time_seconds": scene_end_shot["end_time_seconds"],
            "duration_seconds": scene_end_shot["end_time_seconds"] - scene_start_shot["start_time_seconds"],
            "num_shots": len(current_scene_shots),
            "shots": list(current_scene_shots)
        })

    logger.info(f"Grouped {len(shots_with_features)} shots into {len(scenes)} scenes.")
    return scenes