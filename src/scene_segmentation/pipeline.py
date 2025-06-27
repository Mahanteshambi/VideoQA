# video_analysis_project/src/scene_segmentation/pipeline.py

import logging
import json
import csv
from pathlib import Path
import time
import numpy as np
from typing import Any, Dict, List, Union

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

def write_shot_to_csv(shot: Dict, output_file: Path, fieldnames: List[str], is_first_shot: bool = False) -> None:
    """
    Write a single shot's metadata to CSV file.
    
    Args:
        shot: Shot dictionary with metadata
        output_file: Path to output CSV file
        fieldnames: List of CSV column names
        is_first_shot: Whether this is the first shot (write header if True)
    """
    mode = 'w' if is_first_shot else 'a'
    
    with open(output_file, mode, newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if is_first_shot:
            writer.writeheader()
        
        row = {
            'shot_number': shot.get('shot_number', ''),
            'start_time_seconds': shot.get('start_time_seconds', ''),
            'end_time_seconds': shot.get('end_time_seconds', ''),
            'start_frame': shot.get('start_frame', ''),
            'end_frame': shot.get('end_frame', ''),
            'duration_seconds': shot.get('duration_seconds', ''),
        }
        
        # Extract metadata fields
        vllm_metadata = shot.get('vllm_metadata', {})
        if isinstance(vllm_metadata, str):
            # If metadata is a string (raw response), try to parse it
            try:
                import json
                vllm_metadata = json.loads(vllm_metadata)
            except:
                vllm_metadata = {}
        
        # Map metadata fields to CSV columns
        metadata_mapping = {
            'metadata_ShotDescription': vllm_metadata.get('ShotDescription', ''),
            'metadata_GenreCues': json.dumps(vllm_metadata.get('GenreCues', []), ensure_ascii=False),
            'metadata_SubgenreCues': json.dumps(vllm_metadata.get('SubgenreCues', []), ensure_ascii=False),
            'metadata_AdjectiveTheme': json.dumps(vllm_metadata.get('AdjectiveTheme', []), ensure_ascii=False),
            'metadata_Mood': json.dumps(vllm_metadata.get('Mood', []), ensure_ascii=False),
            'metadata_SettingContext': json.dumps(vllm_metadata.get('SettingContext', []), ensure_ascii=False),
            'metadata_ContentDescriptors': json.dumps(vllm_metadata.get('ContentDescriptors', []), ensure_ascii=False),
            'metadata_LocationHints_Regional': json.dumps(vllm_metadata.get('LocationHints_Regional', []), ensure_ascii=False),
            'metadata_LocationHints_International': json.dumps(vllm_metadata.get('LocationHints_International', []), ensure_ascii=False),
            'metadata_SearchKeywords': json.dumps(vllm_metadata.get('SearchKeywords', []), ensure_ascii=False),
        }
        
        row.update(metadata_mapping)
        writer.writerow(row)
    
    if is_first_shot:
        logger.info(f"Created CSV file and wrote header: {output_file}")
    else:
        logger.debug(f"Wrote shot {shot.get('shot_number', 'unknown')} to CSV")

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
    vllm_annotator_type: str = "smolvlm", # New parameter: "llava_next" or "internvl_chat" or "internvl_3_1b"
    llava_model_checkpoint: str = "llava-hf/llava-next-video-7b-hf", # Specific to LLaVA
    internvl_model_checkpoint: str = "OpenGVLab/InternVL-Chat-V1-5", # Specific to InternVL
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
    shot_metadata_csv_file = video_scene_output_dir / f"{video_name_stem}_shot_metadata.csv"

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
        elif vllm_annotator_type == "smolvlm":
            from .smolvlm_shot_annotator5 import SmolVLMShotAnnotatorV5
            vllm_annotator = SmolVLMShotAnnotatorV5()
        else:
            raise ValueError(f"Invalid VLLM annotator type: {vllm_annotator_type}")
    except ImportError:
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
                # Note: No CSV writing here since cached features should already have a corresponding CSV file
                # from the previous run. Writing again would be redundant and could overwrite external modifications.
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

        # Define CSV fieldnames for incremental writing
        csv_fieldnames = [
            'shot_number', 'start_time_seconds', 'end_time_seconds', 
            'start_frame', 'end_frame', 'duration_seconds',
            'metadata_ShotDescription', 'metadata_GenreCues', 'metadata_SubgenreCues',
            'metadata_AdjectiveTheme', 'metadata_Mood', 'metadata_SettingContext',
            'metadata_ContentDescriptors', 'metadata_LocationHints_Regional',
            'metadata_LocationHints_International', 'metadata_SearchKeywords'
        ]

        feature_extraction_errors = 0
        for i, shot_info_raw in enumerate(shots):
            logger.info(f"Extracting features for shot {shot_info_raw['shot_number']}/{len(shots)}...")
            try:
                shot_features_data = {}
                # 2b. Extract VLLM generative metadata
                vllm_generated_metadata = None
                if vllm_annotator:
                    vllm_generated_metadata = vllm_annotator.extract_metadata_for_shot(
                        original_video_path=video_path,
                        shot_info=shot_info_raw,
                        num_keyframes_to_sample=10 # Number of frames to feed to the generative VLLM
                    )
                
                # Combine original shot_info, similarity features, and VLLM metadata
                combined_shot_data = {
                    **shot_info_raw, 
                    "features": shot_features_data,
                    "vllm_metadata": vllm_generated_metadata if vllm_generated_metadata else {}
                }
                shots_with_features.append(combined_shot_data)
                
                # Write shot to CSV immediately after processing
                try:
                    write_shot_to_csv(
                        combined_shot_data, 
                        shot_metadata_csv_file, 
                        csv_fieldnames, 
                        is_first_shot=(i == 0)
                    )
                except Exception as csv_error:
                    logger.warning(f"Failed to write shot {shot_info_raw['shot_number']} to CSV: {csv_error}")
                    pipeline_results["errors"].append(f"CSV write error for shot {shot_info_raw['shot_number']}: {str(csv_error)}")
                
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
        
        # Cache the features
        try:
            # Convert numpy arrays to lists for JSON serialization
            serializable_features = convert_numpy_to_list(shots_with_features)
            with open(features_cache_file, 'w') as f:
                json.dump(serializable_features, f, indent=2)
            logger.info(f"Cached shot features to {features_cache_file}")
        except Exception as e:
            logger.warning(f"Failed to cache shot features: {e}")

        logger.info(f"Feature extraction completed in {time.time() - start_time_features:.2f}s. {feature_extraction_errors} errors.")

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