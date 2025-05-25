# video_analysis_project/src/scene_segmentation/pipeline.py

import logging
import json
from pathlib import Path
import time
import numpy as np
from typing import Any, Dict, List, Union

from .shot_detector import detect_shots_pyscenedetect
from .feature_extractor import extract_all_features_for_shot
from .scene_grouper import group_shots_into_scenes

# Ensure models are loaded in feature_extractor (they are, at module level)

logger = logging.getLogger(__name__)

def convert_numpy_to_list(obj: Any) -> Any:
    """
    Recursively converts numpy arrays and other non-serializable types to Python native types.
    
    Args:
        obj: Any Python object that might contain numpy arrays
        
    Returns:
        A JSON-serializable version of the object
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int8, np.int16, np.int32, np.int64,
                         np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.complex64, np.complex128)):
        return str(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_list(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_to_list(item) for item in obj]
    elif hasattr(obj, '__dict__'):
        return convert_numpy_to_list(obj.__dict__)
    return obj

def segment_video_into_scenes(
    video_path: str,
    ingestion_output_dir: str, # Base dir where Module 2.1 saved its outputs
    scene_segmentation_output_dir: str, # Base dir for this module's outputs
    shot_detector_threshold: float = 30.0,
    min_shot_len_frames: int = 15,
    num_keyframes_per_shot: int = 1,
    modality_weights: dict = {"visual": 0.6, "audio": 0.1, "textual": 0.3}, # Audio weight low due to placeholder
    scene_similarity_threshold: float = 0.4, # Threshold to break scenes
    min_shots_per_scene: int = 2,
    force_reprocessing: bool = False
) -> dict:
    """
    Orchestrates the full scene segmentation pipeline for a video.
    1. Detects shots.
    2. Extracts multimodal features for each shot.
    3. Groups shots into scenes.

    Args:
        video_path (str): Path to the original video file.
        ingestion_output_dir (str): Path to the directory where video_ingestion module
                                   saved its results (needed for audio file and transcript).
        scene_segmentation_output_dir (str): Directory to save scene segmentation results.
        shot_detector_threshold (float): PySceneDetect ContentDetector threshold.
        min_shot_len_frames (int): Min frames for a shot.
        num_keyframes_per_shot (int): Keyframes for visual feature extraction per shot.
        modality_weights (dict): Weights for visual, audio, textual similarity.
        scene_similarity_threshold (float): Similarity score below which a new scene is started.
        min_shots_per_scene (int): Minimum number of shots to constitute a scene.
        force_reprocessing (bool): If True, reprocesses even if output exists.

    Returns:
        dict: Contains the list of scenes with their shots and features, and other metadata.
    """
    video_p = Path(video_path)
    video_name_stem = video_p.stem
    
    # Prepare output directory for this video's scene segmentation
    video_scene_output_dir = Path(scene_segmentation_output_dir) / video_name_stem
    
    # Results file path
    final_results_file = video_scene_output_dir / f"{video_name_stem}_scene_segmentation_results.json"

    if final_results_file.exists() and not force_reprocessing:
        logger.info(f"Scene segmentation results already exist for {video_name_stem} at {final_results_file}. Skipping.")
        try:
            with open(final_results_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load existing results file {final_results_file}: {e}. Reprocessing.")
    
    video_scene_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Starting scene segmentation pipeline for: {video_p.name}")

    pipeline_results = {
        "video_path": video_path,
        "status": "processing",
        "shot_detection": {"status": "pending", "count": 0, "shots": []},
        "feature_extraction": {"status": "pending", "shots_processed": 0, "errors": 0},
        "scene_grouping": {"status": "pending", "scene_count": 0, "scenes": []},
        "errors": []
    }

    # --- Step 1: Shot Detection ---
    logger.info("--- Running Shot Detection ---")
    start_time_shots = time.time()
    try:
        shots = detect_shots_pyscenedetect(video_path, threshold=shot_detector_threshold, min_scene_len_frames=min_shot_len_frames)
        pipeline_results["shot_detection"]["shots"] = shots
        pipeline_results["shot_detection"]["count"] = len(shots)
        pipeline_results["shot_detection"]["status"] = "completed"
        logger.info(f"Shot detection completed in {time.time() - start_time_shots:.2f}s. Found {len(shots)} shots.")
        if not shots:
            logger.warning(f"No shots detected for {video_p.name}. Cannot proceed with scene segmentation.")
            pipeline_results["status"] = "failed_no_shots"
            pipeline_results["errors"].append("No shots detected.")
            # Save and return
            with open(final_results_file, 'w') as f: json.dump(pipeline_results, f, indent=2)
            return pipeline_results
    except Exception as e:
        logger.error(f"Shot detection failed for {video_p.name}: {e}", exc_info=True)
        pipeline_results["shot_detection"]["status"] = "failed"
        pipeline_results["status"] = "failed_shot_detection"
        pipeline_results["errors"].append(f"Shot detection error: {str(e)}")
        with open(final_results_file, 'w') as f: json.dump(pipeline_results, f, indent=2)
        return pipeline_results

    # --- Step 2: Multimodal Feature Extraction per Shot ---
    logger.info("--- Running Multimodal Feature Extraction for each Shot ---")
    start_time_features = time.time()
    shots_with_features = []
    
    # Paths to Module 2.1 outputs
    ingestion_video_dir = Path(ingestion_output_dir) / video_name_stem
    full_audio_file = ingestion_video_dir / "audio" / f"{video_name_stem}_audio.wav" # Assuming WAV from ingestion
    # Assuming transcript from ingestion is a JSON file with Whisper's segment structure
    transcript_json_file = ingestion_video_dir / "transcripts" / f"{video_name_stem}_audio_transcription.json"
    
    full_transcript_segments_data = []
    if transcript_json_file.exists():
        try:
            with open(transcript_json_file, 'r', encoding='utf-8') as f:
                whisper_output = json.load(f)
                full_transcript_segments_data = whisper_output.get("segments", [])
        except Exception as e:
            logger.warning(f"Could not load transcript segments from {transcript_json_file}: {e}. Textual features will be limited.")
            pipeline_results["errors"].append(f"Transcript loading error: {str(e)}")


    if not full_audio_file.exists():
        logger.warning(f"Full audio file not found at {full_audio_file}. Audio features will be skipped for all shots.")
        pipeline_results["errors"].append(f"Full audio file missing: {full_audio_file}")


    feature_extraction_errors = 0
    for i, shot_info_raw in enumerate(pipeline_results["shot_detection"]["shots"]):
        logger.info(f"Extracting features for shot {shot_info_raw['shot_number']}/{len(pipeline_results['shot_detection']['shots'])}...")
        try:
            # Augment raw shot_info with its features
            shot_features_data = extract_all_features_for_shot(
                shot_info=shot_info_raw,
                original_video_path=video_path, # For on-the-fly keyframe extraction
                full_audio_file_path=str(full_audio_file) if full_audio_file.exists() else None,
                full_transcript_segments=full_transcript_segments_data,
                num_keyframes_for_visual=num_keyframes_per_shot
            )
            # Combine original shot_info with new features
            # Make sure not to overwrite original shot_info keys if they clash with feature keys
            combined_shot_data = {**shot_info_raw, "features": shot_features_data}
            shots_with_features.append(combined_shot_data)
        except Exception as e:
            logger.error(f"Failed to extract features for shot {shot_info_raw['shot_number']}: {e}", exc_info=True)
            feature_extraction_errors += 1
            # Add shot without features or with partial features if needed
            shots_with_features.append({**shot_info_raw, "features": {"error": str(e)}})


    pipeline_results["feature_extraction"]["shots_processed"] = len(shots_with_features)
    pipeline_results["feature_extraction"]["errors"] = feature_extraction_errors
    if feature_extraction_errors > 0:
        pipeline_results["feature_extraction"]["status"] = "completed_with_errors"
    else:
        pipeline_results["feature_extraction"]["status"] = "completed"
    logger.info(f"Feature extraction completed in {time.time() - start_time_features:.2f}s. {feature_extraction_errors} errors.")


    # --- Step 3: Scene Grouping ---
    logger.info("--- Running Scene Grouping ---")
    start_time_grouping = time.time()
    if not shots_with_features or all("error" in s.get("features", {}) for s in shots_with_features):
        logger.error("No shots with features available for scene grouping. Aborting.")
        pipeline_results["status"] = "failed_no_features_for_grouping"
        pipeline_results["errors"].append("Scene grouping aborted due to lack of features.")
    else:
        # Filter out shots where feature extraction might have critically failed if necessary
        valid_shots_for_grouping = [s for s in shots_with_features if "error" not in s.get("features", {})]
        
        if not valid_shots_for_grouping:
            logger.error("No valid shots with features remaining after filtering. Aborting scene grouping.")
            pipeline_results["status"] = "failed_no_valid_features_for_grouping"
            pipeline_results["errors"].append("Scene grouping aborted due to no valid features.")
        else:
            try:
                # Pass the 'features' sub-dictionary to the similarity calculator
                # The scene_grouper expects each item in shots_with_features to contain the actual feature vectors
                # and also the original shot_info like start/end times.
                # Our `extract_all_features_for_shot` returns a dict like `{"visual": ..., "audio": ..., "textual": ...}`
                # The `group_shots_into_scenes` expects each shot dict to have these directly or under a 'features' key.
                # Let's adjust `group_shots_into_scenes` or how we pass data.
                # Assuming `calculate_inter_shot_multimodal_similarity` takes `shot1['features']` and `shot2['features']`
                
                # We need to ensure each element in `valid_shots_for_grouping` is structured correctly
                # for `group_shots_into_scenes` and `calculate_inter_shot_multimodal_similarity`.
                # The current `calculate_inter_shot_multimodal_similarity` expects feature dicts directly.
                # `group_shots_into_scenes` iterates and passes elements to `calculate_inter_shot_multimodal_similarity`.
                # Let's make sure the `valid_shots_for_grouping` elements contain features directly accessible.
                
                # Modifying data structure slightly for `group_shots_into_scenes`
                # It expects the feature vectors themselves to be top-level in the dicts it processes for similarity.
                # Or, the similarity calculator needs to access `shot['features']['visual']` etc.
                # The current `calculate_inter_shot_multimodal_similarity` handles `shot_features.get("visual")`
                # so if `shot1_features` is `shot1['features']`, it's fine.

                scenes = group_shots_into_scenes(
                    shots_with_features=valid_shots_for_grouping, # Each item is a dict with original shot info + a 'features' dict
                    similarity_threshold=scene_similarity_threshold,
                    modality_weights=modality_weights,
                    min_shots_per_scene=min_shots_per_scene
                )
                pipeline_results["scene_grouping"]["scenes"] = scenes
                pipeline_results["scene_grouping"]["scene_count"] = len(scenes)
                pipeline_results["scene_grouping"]["status"] = "completed"
                logger.info(f"Scene grouping completed in {time.time() - start_time_grouping:.2f}s. Found {len(scenes)} scenes.")
            except Exception as e:
                logger.error(f"Scene grouping failed: {e}", exc_info=True)
                pipeline_results["scene_grouping"]["status"] = "failed"
                pipeline_results["errors"].append(f"Scene grouping error: {str(e)}")

    # Final status
    if pipeline_results["errors"] or \
       pipeline_results["shot_detection"]["status"] == "failed" or \
       pipeline_results["feature_extraction"]["status"] == "completed_with_errors" or \
       pipeline_results["feature_extraction"]["status"] == "failed" or \
       pipeline_results["scene_grouping"]["status"] == "failed":
        pipeline_results["status"] = "pipeline_completed_with_errors"
    elif pipeline_results["shot_detection"]["status"] == "completed" and \
         pipeline_results["feature_extraction"]["status"] == "completed" and \
         pipeline_results["scene_grouping"]["status"] == "completed":
        pipeline_results["status"] = "pipeline_completed_successfully"
    else: # Default to with errors if not explicitly successful
        pipeline_results["status"] = "pipeline_completed_with_errors"


    # Save final results
    try:
        # Clean up NumPy arrays in features for JSON serialization if they are still there
        # (they should be fine as `extract_all_features_for_shot` returns them,
        # and `group_shots_into_scenes` just passes them around.
        # JSON can't serialize numpy arrays directly)
        serializable_results = convert_numpy_to_list(pipeline_results)
        with open(final_results_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved scene segmentation results to {final_results_file}")
        return serializable_results
    except Exception as e:
        error_msg = f"Could not save final scene segmentation results: {str(e)}"
        logger.error(error_msg)
        # Return a simplified version of results if serialization failed
        simplified_results = {
            "status": "error_saving_results",
            "error": error_msg,
            "video_path": video_path,
            "shot_detection": {
                "status": pipeline_results["shot_detection"]["status"],
                "count": pipeline_results["shot_detection"]["count"]
            }
        }
        return simplified_results


    return pipeline_results