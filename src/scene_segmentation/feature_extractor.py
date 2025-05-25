# video_analysis_project/src/scene_segmentation/feature_extractor.py

import torch
import clip # From openai-clip
from PIL import Image
from sentence_transformers import SentenceTransformer
import numpy as np
import librosa # For basic audio features
from pathlib import Path
import logging
import moviepy.editor as mp # For extracting audio segments

logger = logging.getLogger(__name__)

# --- Device Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"FeatureExtractor: Using device: {DEVICE}")

# --- Load Models (Load them once globally or pass them around) ---
# Visual Model (CLIP)
try:
    CLIP_MODEL_NAME = "ViT-B/32"
    clip_model, clip_preprocess = clip.load(CLIP_MODEL_NAME, device=DEVICE, jit=False)
    clip_model.eval() # Set to evaluation mode
    logger.info(f"CLIP model '{CLIP_MODEL_NAME}' loaded successfully for visual features.")
except Exception as e:
    logger.error(f"Failed to load CLIP model: {e}. Visual features will not be available.", exc_info=True)
    clip_model = None
    clip_preprocess = None

# Textual Model (SentenceTransformer)
try:
    TEXT_MODEL_NAME = 'all-MiniLM-L6-v2' # Fast and good quality
    text_model = SentenceTransformer(TEXT_MODEL_NAME, device=DEVICE)
    logger.info(f"SentenceTransformer model '{TEXT_MODEL_NAME}' loaded successfully for textual features.")
except Exception as e:
    logger.error(f"Failed to load SentenceTransformer model: {e}. Textual features will not be available.", exc_info=True)
    text_model = None


def get_visual_features_for_shot(
    shot_info: dict, 
    video_frames_base_dir: str | Path, 
    num_keyframes: int = 3
) -> np.ndarray | None:
    """
    Extracts CLIP visual features for a given shot by sampling keyframes.

    Args:
        shot_info (dict): Dictionary containing 'start_frame', 'end_frame'.
        video_frames_base_dir (str | Path): Base directory where individual frames
                                          (from Module 2.1) are stored. Frame filenames
                                          should allow easy lookup by frame number.
                                          e.g., frame_000001.png, frame_000002.png etc.
                                          OR path to the original video to extract frames on the fly.
                                          For now, assumes frames are pre-extracted.
        num_keyframes (int): Number of keyframes to sample from the shot.

    Returns:
        np.ndarray | None: Averaged CLIP feature vector for the shot, or None if failed.
    """
    if not clip_model or not clip_preprocess:
        logger.warning("CLIP model not available. Skipping visual feature extraction.")
        return None

    start_frame = shot_info["start_frame"]
    end_frame = shot_info["end_frame"] # Assuming this is the last frame IN the shot
    duration_frames = end_frame - start_frame + 1

    if duration_frames <= 0:
        logger.warning(f"Shot {shot_info.get('shot_number', 'N/A')} has no duration in frames. Skipping visual features.")
        return None

    keyframe_features_list = []
    
    # Smarter keyframe selection: equally spaced within the shot
    # Ensure at least one keyframe if num_keyframes > 0
    if num_keyframes == 1:
        frame_indices_to_sample = [start_frame + duration_frames // 2] # Middle frame
    elif duration_frames < num_keyframes: # If shot is shorter than num_keyframes, sample all frames
        frame_indices_to_sample = range(start_frame, end_frame + 1)
    else:
        frame_indices_to_sample = np.linspace(start_frame, end_frame, num_keyframes, dtype=int)

    # This part assumes frames are named like 'frame_000000_time_0.000s.png' etc.
    # And that the frame number in the filename corresponds to the actual frame index.
    # This is a simplification. Robust frame finding might be needed.
    # For now, we assume `video_frames_base_dir` contains appropriately named frames.
    # A better approach might be to pass the video file and extract keyframes on the fly.
    # Let's assume a convention: `frame_{frame_number:06d}.png` for simplicity
    # The frame paths extracted by `video_ingestion.frame_processing` are more complex with timestamps.
    # For now, let's assume we get a list of relevant frame paths for the shot.

    # TODO: This function needs access to the actual frame *files* for the shot.
    # The current `video_frames_base_dir` and frame indexing is a placeholder.
    # A more robust way is to pass the original video path and extract needed frames here.
    # OR, the orchestrator (pipeline.py) should provide paths to specific frames for this shot.

    # --- Simplified Placeholder for getting frame paths for a shot ---
    # This requires Module 2.1 to output frames in a predictable way, OR we re-extract here.
    # For now, let's assume a function `get_frame_paths_for_shot` exists or is handled by the caller.
    # If `video_frames_base_dir` is the original video path:
    temp_frame_paths_to_load = []
    try:
        video_clip = mp.VideoFileClip(str(video_frames_base_dir)) # Assuming it's the video path
        for frame_idx in frame_indices_to_sample:
            # Convert frame index to time; assumes constant frame rate from video metadata
            # This is tricky if original FPS isn't easily available here.
            # A better approach: if the shot_info contains start/end time_seconds:
            time_in_seconds_samples = np.linspace(shot_info["start_time_seconds"], shot_info["end_time_seconds"], num_keyframes)
            
            for i, t in enumerate(time_in_seconds_samples):
                temp_frame_file = Path(f"temp_shot_frame_{shot_info['shot_number']}_{i}.png")
                video_clip.save_frame(str(temp_frame_file), t=t)
                temp_frame_paths_to_load.append(temp_frame_file)
        video_clip.close()
    except Exception as e_vid:
        logger.error(f"Could not extract temp keyframes for shot {shot_info.get('shot_number', 'N/A')}: {e_vid}")
        return None
    # --- End Simplified Placeholder ---

    if not temp_frame_paths_to_load:
        logger.warning(f"No keyframes found/extracted for shot {shot_info.get('shot_number', 'N/A')}.")
        return None

    for frame_path in temp_frame_paths_to_load:
        try:
            if not Path(frame_path).exists():
                logger.warning(f"Keyframe path {frame_path} does not exist for shot {shot_info.get('shot_number', 'N/A')}.")
                continue
            image = Image.open(frame_path)
            image_input = clip_preprocess(image).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                image_features = clip_model.encode_image(image_input)
            keyframe_features_list.append(image_features.squeeze().cpu().numpy())
        except Exception as e:
            logger.error(f"Error processing keyframe {frame_path} for shot {shot_info.get('shot_number', 'N/A')}: {e}")
        finally:
            if Path(frame_path).exists() and "temp_shot_frame" in str(frame_path): # Clean up temp frames
                Path(frame_path).unlink()


    if not keyframe_features_list:
        logger.warning(f"No visual features extracted for shot {shot_info.get('shot_number', 'N/A')}.")
        return None

    # Average the features of the keyframes
    avg_features = np.mean(keyframe_features_list, axis=0)
    return avg_features / np.linalg.norm(avg_features) # Normalize


def get_audio_features_for_shot(
    shot_info: dict,
    full_audio_file_path: str | Path,
    sample_rate: int = 16000 # Common for audio models
) -> np.ndarray | None:
    """
    Extracts basic audio features (e.g., mean RMS energy) for a given shot's audio segment.
    This is a placeholder for more sophisticated audio embeddings.
    """
    full_audio_file_path = Path(full_audio_file_path)
    if not full_audio_file_path.exists():
        logger.warning(f"Full audio file {full_audio_file_path} not found. Skipping audio features for shot.")
        return None

    start_sec = shot_info["start_time_seconds"]
    duration_sec = shot_info["duration_seconds"]

    if duration_sec <= 0:
        logger.warning(f"Shot {shot_info.get('shot_number', 'N/A')} has no audio duration. Skipping audio features.")
        return None

    try:
        # Load only the segment for the shot
        y_shot, sr_native = librosa.load(
            str(full_audio_file_path), 
            sr=sample_rate,       # Resample to target rate
            offset=start_sec, 
            duration=duration_sec
        )
        
        if len(y_shot) == 0:
            logger.warning(f"No audio data loaded for shot {shot_info.get('shot_number', 'N/A')} segment. Might be too short or an issue with offsets.")
            return np.array([0.0]) # Return a default neutral feature

        # Example: RMS energy (simple placeholder)
        rms_energy = librosa.feature.rms(y=y_shot)[0]
        mean_rms_energy = np.mean(rms_energy)
        
        # Placeholder feature vector (just one value for now)
        # In future, replace with actual audio embeddings (e.g., VGGish, YAMNet)
        # which would be much higher dimensional and more semantic.
        audio_feature_vector = np.array([mean_rms_energy]) 
        # Normalize if it makes sense for the feature type
        # For RMS, it's already positive. If combining with cosine sims, scale might matter.
        return audio_feature_vector / (np.linalg.norm(audio_feature_vector) + 1e-6) if np.linalg.norm(audio_feature_vector) > 0 else audio_feature_vector

    except Exception as e:
        logger.error(f"Error extracting audio features for shot {shot_info.get('shot_number', 'N/A')}: {e}")
        return None


def get_textual_features_for_shot(
    shot_info: dict,
    full_transcript_data: list[dict] # List of whisper segments: [{'start': float, 'end': float, 'text': str}]
) -> np.ndarray | None:
    """
    Extracts SentenceTransformer textual features for a given shot from full transcript.
    """
    if not text_model:
        logger.warning("SentenceTransformer model not available. Skipping textual feature extraction.")
        return None
    if not full_transcript_data:
        logger.info(f"No transcript data provided for shot {shot_info.get('shot_number', 'N/A')}. Skipping textual features.")
        return None

    shot_start_sec = shot_info["start_time_seconds"]
    shot_end_sec = shot_info["end_time_seconds"]
    
    relevant_texts = []
    for segment in full_transcript_data:
        seg_start = segment.get("start")
        seg_end = segment.get("end")
        seg_text = segment.get("text", "").strip()

        if seg_start is None or seg_end is None or not seg_text:
            continue

        # Check for overlap: if segment midpoint is within shot, or segment overlaps significantly
        # A simple heuristic: segment midpoint falls within the shot
        seg_midpoint = (seg_start + seg_end) / 2
        if shot_start_sec <= seg_midpoint < shot_end_sec:
            relevant_texts.append(seg_text)
        # More robust overlap: (max(shot_start, seg_start) < min(shot_end, seg_end))
        elif max(shot_start_sec, seg_start) < min(shot_end_sec, seg_end):
             relevant_texts.append(seg_text)


    if not relevant_texts:
        logger.info(f"No relevant transcript segments found for shot {shot_info.get('shot_number', 'N/A')} ({shot_start_sec:.2f}s - {shot_end_sec:.2f}s).")
        # Return a zero vector or some other placeholder for "no text"
        # text_embedding_dim = text_model.get_sentence_embedding_dimension()
        # return np.zeros(text_embedding_dim)
        return None # Or a specific "no text" embedding

    combined_text = " ".join(relevant_texts)
    
    try:
        text_embedding = text_model.encode(combined_text, convert_to_numpy=True, show_progress_bar=False)
        return text_embedding / np.linalg.norm(text_embedding) # Normalize
    except Exception as e:
        logger.error(f"Error encoding text for shot {shot_info.get('shot_number', 'N/A')}: {e}")
        return None

def extract_all_features_for_shot(
    shot_info: dict,
    original_video_path: str, # Needed for on-the-fly frame/audio extraction
    # video_frames_dir: str, # No longer primary if extracting on the fly
    full_audio_file_path: str, # From Module 2.1
    full_transcript_segments: list[dict], # From Module 2.1 (Whisper's segment output)
    num_keyframes_for_visual: int = 1
) -> dict:
    """
    Orchestrates extraction of visual, audio, and textual features for a single shot.
    """
    logger.debug(f"Extracting all features for shot {shot_info['shot_number']} ({shot_info['start_time_seconds']:.2f}s - {shot_info['end_time_seconds']:.2f}s)")
    
    features = {
        "shot_number": shot_info["shot_number"],
        "visual": None,
        "audio": None,
        "textual": None
    }

    # Visual features (pass original video path to extract keyframes from)
    # This assumes get_visual_features_for_shot can handle original_video_path
    features["visual"] = get_visual_features_for_shot(
        shot_info, 
        original_video_path, # Pass video path instead of frames_dir
        num_keyframes=num_keyframes_for_visual
    )
    
    # Audio features
    features["audio"] = get_audio_features_for_shot(shot_info, full_audio_file_path)
    
    # Textual features
    features["textual"] = get_textual_features_for_shot(shot_info, full_transcript_segments)
    
    return features