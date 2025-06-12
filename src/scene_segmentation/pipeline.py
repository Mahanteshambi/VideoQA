# video_analysis_project/src/scene_segmentation/pipeline.py

import logging
import json
from pathlib import Path
import time
import numpy as np
from typing import Any, Dict, List, Union
import sys
import os

# Handle imports differently depending on how the file is run
if __name__ == "__main__":
    # When run as a script
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from src.scene_segmentation.shot_detector import detect_shots_pyscenedetect
    from src.scene_segmentation.feature_extractor import extract_all_features_for_shot
    from src.scene_segmentation.scene_grouper import group_shots_into_scenes
else:
    # When imported as a module
    from .shot_detector import detect_shots_pyscenedetect
    from .feature_extractor import extract_all_features_for_shot
    from .scene_grouper import group_shots_into_scenes

# Ensure models are loaded in feature_extractor (they are, at module level)

logger = logging.getLogger(__name__)

internvl_3_1b_model_checkpoint = "OpenGVLab/InternVL3-1B"
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
    modality_weights: dict = {"visual": 0.9, "audio": 0.0, "textual": 0.1}, # Audio weight low due to placeholder
    scene_similarity_threshold: float = 0.4, # Threshold to break scenes
    min_shots_per_scene: int = 2,
    shotdetection_reprocessing: bool = False,
    feature_extraction_reprocessing: bool = False,
    vllm_annotator_type: str = "internvl_3_1b", # New parameter: "llava_next" or "internvl_chat" or "internvl_3_1b"
    llava_model_checkpoint: str = "llava-hf/llava-next-video-7b-hf", # Specific to LLaVA
    internvl_model_checkpoint: str = "OpenGVLab/InternVL-Chat-V1-5", # Specific to InternVL
    batch_size: int = 8,
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
    video_scene_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Results file paths
    shots_cache_file = video_scene_output_dir / f"{video_name_stem}_shots.json"
    features_cache_file = video_scene_output_dir / f"{video_name_stem}_{vllm_annotator_type}_shot_features.json"
    scenes_output_file = video_scene_output_dir / f"{video_name_stem}_scenes.json"
    pipeline_status_file = video_scene_output_dir / f"{video_name_stem}_pipeline_status.json"

    pipeline_results = {
        "video_path": video_path,
        "status": "processing",
        "shot_detection": {"status": "pending", "count": 0},
        "feature_extraction": {"status": "pending", "shots_processed": 0, "errors": 0},
        "scene_grouping": {"status": "pending", "scene_count": 0},
        "errors": []
    }
    
    # Initialize VLLM annotator (optional)
    vllm_annotator = None
    try:
        if vllm_annotator_type == "llava_next":
            from .llava_next_shot_annotator import LlavaNextShotAnnotator
            vllm_annotator = LlavaNextShotAnnotator(model_checkpoint=llava_model_checkpoint)
        elif vllm_annotator_type == "internvl_chat":
            from .internvl_chat_shot_annotator import InternVLChatShotAnnotator
            vllm_annotator = InternVLChatShotAnnotator(model_checkpoint=internvl_model_checkpoint)
        elif vllm_annotator_type == "internvl_3_1b":
            from .internvl_3_1b_shot_annotator import InternVL3_1B_ShotAnnotator
            vllm_annotator = InternVL3_1B_ShotAnnotator(model_checkpoint=internvl_3_1b_model_checkpoint)
        else:
            raise ValueError(f"Invalid VLLM annotator type: {vllm_annotator_type}")
    except ImportError as e_import_vllm:
        logger.warning(f"VLLM annotator module not available. VLLM metadata will be skipped. Error: {e_import_vllm}")
        logger.warning("VLLM annotator module not available. VLLM metadata will be skipped.")
    except Exception as e_init_vllm:
        logger.error(f"Failed to initialize VLLMShotAnnotator: {e_init_vllm}. VLLM metadata will be skipped.", exc_info=True)
        pipeline_results["errors"].append(f"VLLM Annotator Init Error: {str(e_init_vllm)}")

    # --- Step 1: Shot Detection ---
    logger.info("--- Running Shot Detection ---")
    start_time_shots = time.time()
    
    # Try to load cached shots first
    shots = None
    if shots_cache_file.exists() and not shotdetection_reprocessing:
        try:
            logger.info(f"Found cached shot detection results at {shots_cache_file}")
            with open(shots_cache_file, 'r') as f:
                shots = json.load(f)
            pipeline_results["shot_detection"]["count"] = len(shots)
            pipeline_results["shot_detection"]["status"] = "completed_from_cache"
            logger.info(f"Loaded {len(shots)} shots from cache.")
        except Exception as e:
            logger.warning(f"Failed to load cached shots from {shots_cache_file}: {e}. Will redetect shots.")
            shots = None

    # Run shot detection if needed
    if shots is None:
        try:
            shots = detect_shots_pyscenedetect(video_path, threshold=shot_detector_threshold, min_scene_len_frames=min_shot_len_frames)
            pipeline_results["shot_detection"]["count"] = len(shots)
            pipeline_results["shot_detection"]["status"] = "completed"
            logger.info(f"Shot detection completed in {time.time() - start_time_shots:.2f}s. Found {len(shots)} shots.")
            
            # Cache the shot detection results
            try:
                with open(shots_cache_file, 'w') as f:
                    json.dump(shots, f, indent=2)
                logger.info(f"Cached shot detection results to {shots_cache_file}")
            except Exception as e:
                logger.warning(f"Failed to cache shot detection results: {e}")
        except Exception as e:
            logger.error(f"Shot detection failed for {video_p.name}: {e}", exc_info=True)
            pipeline_results["shot_detection"]["status"] = "failed"
            pipeline_results["status"] = "failed_shot_detection"
            pipeline_results["errors"].append(f"Shot detection error: {str(e)}")
            _save_pipeline_status(pipeline_status_file, pipeline_results)
            return pipeline_results

    if not shots:
        logger.warning(f"No shots detected for {video_p.name}. Cannot proceed with scene segmentation.")
        pipeline_results["status"] = "failed_no_shots"
        pipeline_results["errors"].append("No shots detected.")
        _save_pipeline_status(pipeline_status_file, pipeline_results)
        return pipeline_results

    # --- Step 2: Multimodal Feature Extraction per Shot ---
    logger.info("--- Running Multimodal Feature Extraction for each Shot ---")
    start_time_features = time.time()
    shots_with_features = []
    # Cache file for final metadata results
    metadata_cache_file = video_scene_output_dir / f"{video_name_stem}_{vllm_annotator_type}_shot_metadata.json"
    
    # Try to load cached features first
    cached_features = None
    if features_cache_file.exists() and not feature_extraction_reprocessing:
        try:
            logger.info(f"Found cached shot features at {features_cache_file}")
            with open(features_cache_file, 'r') as f:
                cached_features = json.load(f)
            
            # Verify the cached features match our current shots
            if len(cached_features) == len(shots) and all(
                cf.get('shot_number') == shot.get('shot_number') 
                and cf.get('start_time_seconds') == shot.get('start_time_seconds')
                and cf.get('end_time_seconds') == shot.get('end_time_seconds')
                for cf, shot in zip(cached_features, shots)
            ):
                logger.info(f"Using cached features for {len(shots)} shots")
                shots_with_features = cached_features
                pipeline_results["feature_extraction"]["status"] = "completed_from_cache"
                pipeline_results["feature_extraction"]["shots_processed"] = len(shots)
                pipeline_results["feature_extraction"]["errors"] = 0
            else:
                logger.warning("Cached features don't match current shots. Will recompute features.")
                cached_features = None
        except Exception as e:
            logger.warning(f"Failed to load cached features from {features_cache_file}: {e}. Will recompute features.")
            cached_features = None

    # Extract features if needed
    if cached_features is None:
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

        logger.info("--- Running Batched VLLM Metadata Generation ---")
        start_time_metadata = time.time()
        
        # Try to load cached results first
        if metadata_cache_file.exists() and not feature_extraction_reprocessing:
            logger.info(f"Found cached metadata at {metadata_cache_file}, skipping generation.")
            with open(metadata_cache_file, 'r') as f:
                shots_with_metadata = json.load(f)
            return {"status": "completed_from_cache", "results": shots_with_metadata}

        shots_with_metadata = []
        metadata_errors = 0

        # ## MODIFIED ##: This is the new batched loop.
        for i in range(0, len(shots), batch_size):
            shot_batch_info = shots[i : i + batch_size]
            logger.info(f"Processing batch of {len(shot_batch_info)} shots (starting with shot #{shot_batch_info[0]['shot_number']})...")

            try:
                # Call the new batch method in the annotator
                vllm_metadata_batch = vllm_annotator.extract_metadata_for_batch(
                    original_video_path=video_path,
                    shots_info=shot_batch_info,
                    num_keyframes_to_sample=8 # You can tune this
                )

                # Combine results with original shot info
                for shot_info, metadata in zip(shot_batch_info, vllm_metadata_batch):
                    if "error" in metadata:
                        metadata_errors += 1
                    shots_with_metadata.append({**shot_info, "vllm_metadata": metadata})

            except Exception as e:
                logger.error(f"A critical error occurred processing batch starting at shot {i}: {e}", exc_info=True)
                metadata_errors += len(shot_batch_info)
                # Add error placeholders for all shots in the failed batch
                for shot_info in shot_batch_info:
                    shots_with_metadata.append({**shot_info, "vllm_metadata": {"error": f"Critical batch failure: {e}"}})

        logger.info(f"Metadata generation completed in {time.time() - start_time_metadata:.2f}s. Found {metadata_errors} errors.")
        
        # Cache the final results
        try:
            with open(metadata_cache_file, 'w') as f:
                json.dump(shots_with_metadata, f, indent=2)
            logger.info(f"Cached shot metadata to {metadata_cache_file}")
        except Exception as e:
            logger.warning(f"Failed to cache shot metadata: {e}")

        # Set shots_with_features from shots_with_metadata
        shots_with_features = shots_with_metadata
        
        # Update pipeline results
        pipeline_results["feature_extraction"]["shots_processed"] = len(shots_with_features)
        pipeline_results["feature_extraction"]["errors"] = metadata_errors
        if metadata_errors > 0:
            pipeline_results["feature_extraction"]["status"] = "completed_with_errors"
        else:
            pipeline_results["feature_extraction"]["status"] = "completed"
        
        # Cache the features
        try:
            # Convert numpy arrays to lists for JSON serialization
            serializable_features = convert_numpy_to_list(shots_with_features)
            with open(features_cache_file, 'w') as f:
                json.dump(serializable_features, f, indent=2)
            logger.info(f"Cached shot features to {features_cache_file}")
        except Exception as e:
            logger.warning(f"Failed to cache shot features: {e}")

        logger.info(f"Feature extraction completed in {time.time() - start_time_features:.2f}s. {metadata_errors} errors.")

    # --- Step 3: Scene Grouping ---
    logger.info("--- Running Scene Grouping ---")
    start_time_grouping = time.time()
    if not shots_with_features or all("error" in s.get("vllm_metadata", {}) for s in shots_with_features):
        logger.error("No shots with features available for scene grouping. Aborting.")
        pipeline_results["status"] = "failed_no_features_for_grouping"
        pipeline_results["errors"].append("Scene grouping aborted due to lack of features.")
        _save_pipeline_status(pipeline_status_file, pipeline_results)
    else:
        # Filter out shots where feature extraction might have critically failed if necessary
        valid_shots_for_grouping = [s for s in shots_with_features if "error" not in s.get("vllm_metadata", {})]
        
        if not valid_shots_for_grouping:
            logger.error("No valid shots with features remaining after filtering. Aborting scene grouping.")
            pipeline_results["status"] = "failed_no_valid_features_for_grouping"
            pipeline_results["errors"].append("Scene grouping aborted due to no valid features.")
            _save_pipeline_status(pipeline_status_file, pipeline_results)
        else:
            try:
                scenes = group_shots_into_scenes(
                    shots_with_features=valid_shots_for_grouping,
                    similarity_threshold=scene_similarity_threshold,
                    modality_weights=modality_weights,
                    min_shots_per_scene=min_shots_per_scene
                )
                pipeline_results["scene_grouping"]["scene_count"] = len(scenes)
                pipeline_results["scene_grouping"]["status"] = "completed"
                logger.info(f"Scene grouping completed in {time.time() - start_time_grouping:.2f}s. Found {len(scenes)} scenes.")

                # Save scenes to separate file
                try:
                    serializable_scenes = convert_numpy_to_list(scenes)
                    with open(scenes_output_file, 'w') as f:
                        json.dump(serializable_scenes, f, indent=2)
                    logger.info(f"Saved scene segmentation results to {scenes_output_file}")
                except Exception as e:
                    logger.error(f"Failed to save scenes to file: {e}")
                    pipeline_results["errors"].append(f"Scene saving error: {str(e)}")

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
    elif pipeline_results["shot_detection"]["status"] in ["completed", "completed_from_cache"] and \
         pipeline_results["feature_extraction"]["status"] in ["completed", "completed_from_cache"] and \
         pipeline_results["scene_grouping"]["status"] == "completed":
        pipeline_results["status"] = "pipeline_completed_successfully"
    else: # Default to with errors if not explicitly successful
        pipeline_results["status"] = "pipeline_completed_with_errors"

    # Save final pipeline status
    _save_pipeline_status(pipeline_status_file, pipeline_results)
    return pipeline_results

def _save_pipeline_status(status_file: Path, status: dict) -> None:
    """Helper function to save pipeline status to file."""
    try:
        with open(status_file, 'w') as f:
            json.dump(status, f, indent=2)
        logger.info(f"Saved pipeline status to {status_file}")
    except Exception as e:
        logger.error(f"Failed to save pipeline status: {e}")

# Add a simple main function for demonstration when run directly
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run video scene segmentation pipeline')
    parser.add_argument('--video_path', type=str, required=True, help='Path to the input video file')
    parser.add_argument('--ingestion_output_dir', type=str, required=True, help='Path to the ingestion output directory')
    parser.add_argument('--scene_segmentation_output_dir', type=str, required=True, help='Path to the scene segmentation output directory')
    
    args = parser.parse_args()
    
    print(f"Starting scene segmentation pipeline for video: {args.video_path}")
    result = segment_video_into_scenes(
        video_path=args.video_path,
        ingestion_output_dir=args.ingestion_output_dir,
        scene_segmentation_output_dir=args.scene_segmentation_output_dir
    )
    print(f"Pipeline completed with status: {result['status']}")