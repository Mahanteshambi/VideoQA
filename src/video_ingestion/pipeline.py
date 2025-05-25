# video_analysis_project/src/video_ingestion/pipeline.py

from pathlib import Path
import json
import logging

# Relative imports from within the same package
from .utils import check_ffmpeg_availability
from .frame_processing import extract_frames
from .audio_processing import extract_audio, transcribe_audio
from .metadata_extraction import get_video_metadata

logger = logging.getLogger(__name__)
# Configure logging for the pipeline specifically if needed, or rely on root logger
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')


def process_video_pipeline(
    video_path: str, 
    base_output_dir: str, 
    frame_rate: int = 1, 
    whisper_model: str = "base",
    skip_frames: bool = False,
    skip_audio_extraction: bool = False, # Renamed for clarity
    skip_transcription: bool = False,
    skip_metadata: bool = False,
    force_reprocessing: bool = False # New flag
) -> dict:
    """
    Main orchestrator function to process a single video file using modular components.
    Creates a structured output directory for the processed video.
    """
    try:
        check_ffmpeg_availability()
    except FileNotFoundError as e:
        logger.critical(f"Critical FFmpeg/ffprobe availability error: {e}")
        return {"error": str(e), "status": "failed_setup"}

    video_path_obj = Path(video_path)
    if not video_path_obj.is_file():
        logger.error(f"Video file {video_path} not found or is not a file.")
        return {"error": f"Video file {video_path} not found.", "status": "failed_input_video"}

    video_name_stem = video_path_obj.stem
    processed_video_dir = Path(base_output_dir) / video_name_stem
    
    # Check if already processed and force_reprocessing is False
    # A simple check could be if a "processing_complete.json" or similar exists.
    # For now, we'll just check if the main directory exists and skip if not forcing.
    if processed_video_dir.exists() and not force_reprocessing:
        logger.info(f"Output directory {processed_video_dir} already exists. Assuming already processed. Skipping.")
        # Optionally, load results from a previously saved file here
        return {
            "status": "skipped_existing",
            "original_video_path": str(video_path_obj),
            "processed_video_dir": str(processed_video_dir),
            "message": "Output directory exists, skipped reprocessing. Use force_reprocessing=True to override."
        }
        
    processed_video_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Processing video: {video_path_obj.name}. Outputting to: {processed_video_dir}")

    results = {
        "status": "processing", # Will be updated to "completed" or "completed_with_errors"
        "original_video_path": str(video_path_obj),
        "processed_video_dir": str(processed_video_dir),
        "frames_dir": None, "extracted_frames_count": 0,
        "audio_file_path": None,
        "transcript_dir": None, "transcription_json_path": None, "transcription_txt_path": None,
        "metadata": None,
        "errors": []
    }

    # 1. Extract Metadata
    if not skip_metadata:
        logger.info(f"--- Step 1: Extracting Metadata for {video_name_stem} ---")
        try:
            metadata_dict = get_video_metadata(video_path)
            if metadata_dict:
                results["metadata"] = metadata_dict
                # Save metadata to a JSON file
                metadata_file_path = processed_video_dir / f"{video_name_stem}_metadata.json"
                with open(metadata_file_path, 'w', encoding='utf-8') as f_meta:
                    json.dump(metadata_dict, f_meta, indent=2, ensure_ascii=False)
                logger.info(f"Metadata saved to {metadata_file_path}")
            else:
                results["errors"].append("Metadata extraction returned None.")
        except Exception as e:
            err_msg = f"Failed metadata extraction: {e}"
            logger.error(err_msg, exc_info=True)
            results["errors"].append(err_msg)
    else:
        logger.info(f"--- Step 1: Skipping Metadata Extraction for {video_name_stem} ---")

    # 2. Extract Frames
    if not skip_frames:
        frames_output_dir = processed_video_dir / "frames"
        results["frames_dir"] = str(frames_output_dir) # Store dir even if extraction fails
        logger.info(f"--- Step 2: Extracting Frames for {video_name_stem} ---")
        try:
            extracted_paths = extract_frames(video_path, str(frames_output_dir), frame_rate)
            results["extracted_frames_count"] = len(extracted_paths)
        except Exception as e:
            err_msg = f"Failed frame extraction: {e}"
            logger.error(err_msg, exc_info=True)
            results["errors"].append(err_msg)
    else:
        logger.info(f"--- Step 2: Skipping Frame Extraction for {video_name_stem} ---")

    # 3. Extract Audio
    audio_file_path_str = None # Variable to hold the successful audio path
    if not skip_audio_extraction:
        audio_output_dir = processed_video_dir / "audio"
        audio_output_dir.mkdir(parents=True, exist_ok=True)
        # Using .wav for transcription as it's lossless and Whisper handles it well.
        target_audio_file = audio_output_dir / f"{video_name_stem}_audio.wav"
        logger.info(f"--- Step 3: Extracting Audio for {video_name_stem} ---")
        try:
            audio_file_path_str = extract_audio(video_path, str(target_audio_file))
            results["audio_file_path"] = audio_file_path_str
            if not audio_file_path_str: # If extraction function returned None
                results["errors"].append("Audio extraction failed to produce a file.")
        except Exception as e:
            err_msg = f"Failed audio extraction: {e}"
            logger.error(err_msg, exc_info=True)
            results["errors"].append(err_msg)
            results["audio_file_path"] = None # Ensure it's None
    else:
        logger.info(f"--- Step 3: Skipping Audio Extraction for {video_name_stem} ---")
        if not skip_transcription:
            logger.warning("Skipping transcription as audio extraction is skipped.")
            skip_transcription = True

    # 4. Transcribe Audio
    if not skip_transcription:
        if audio_file_path_str and Path(audio_file_path_str).exists():
            transcript_output_dir = processed_video_dir / "transcripts"
            results["transcript_dir"] = str(transcript_output_dir) # Store dir
            logger.info(f"--- Step 4: Transcribing Audio for {video_name_stem} ---")
            try:
                json_path, txt_path = transcribe_audio(audio_file_path_str, str(transcript_output_dir), whisper_model)
                results["transcription_json_path"] = json_path
                results["transcription_txt_path"] = txt_path
                if not json_path: # If transcription returned None for paths
                     results["errors"].append("Audio transcription failed to produce output files.")
            except Exception as e:
                err_msg = f"Failed audio transcription: {e}"
                logger.error(err_msg, exc_info=True)
                results["errors"].append(err_msg)
        else:
            err_msg = "Skipping transcription because audio file was not successfully extracted or path is invalid."
            logger.warning(err_msg)
            if not skip_audio_extraction: # Only add to errors if audio wasn't explicitly skipped
                results["errors"].append(err_msg)
    else:
        logger.info(f"--- Step 4: Skipping Audio Transcription for {video_name_stem} ---")
        
    logger.info(f"--- Processing finished for {video_name_stem} ---")
    if results["errors"]:
        results["status"] = "completed_with_errors"
        logger.error(f"Video processing for {video_name_stem} completed with {len(results['errors'])} errors.")
    else:
        results["status"] = "completed_successfully"
        logger.info(f"Video processing for {video_name_stem} completed successfully.")
        
    # Save a summary of results to a JSON file
    results_summary_path = processed_video_dir / f"{video_name_stem}_processing_summary.json"
    try:
        with open(results_summary_path, 'w', encoding='utf-8') as f_summary:
            json.dump(results, f_summary, indent=2, ensure_ascii=False)
        logger.info(f"Processing summary saved to {results_summary_path}")
    except Exception as e_json:
        logger.error(f"Could not save processing summary for {video_name_stem}: {e_json}", exc_info=True)

    return results