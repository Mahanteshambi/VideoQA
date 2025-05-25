# video_analysis_project/src/video_ingestion/frame_processing.py

import moviepy.editor as mp
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def extract_frames(video_path: str, output_dir: str, frame_rate: int = 1) -> list[str]:
    """
    Extracts frames from a video file using MoviePy.
    (Content is similar to the previous version, just moved to its own file)
    """
    video_path_obj = Path(video_path)
    if not video_path_obj.exists():
        logger.error(f"Video file not found: {video_path}")
        raise FileNotFoundError(f"Video file not found: {video_path}")

    output_dir_obj = Path(output_dir)
    output_dir_obj.mkdir(parents=True, exist_ok=True)

    extracted_frames_paths = []
    try:
        logger.info(f"Starting frame extraction for {video_path_obj.name} at {frame_rate} fps...")
        video_clip = mp.VideoFileClip(video_path)
        duration = video_clip.duration
        
        if duration is None or duration <=0:
            logger.warning(f"Video {video_path_obj.name} has invalid or zero duration. Skipping frame extraction.")
            video_clip.close()
            return []

        for i, frame_time_numerator in enumerate(range(0, int(duration * frame_rate))):
            time_in_seconds = frame_time_numerator / frame_rate
            # Ensure time_in_seconds does not exceed video duration, especially with rounding
            if time_in_seconds > duration:
                break
            
            frame_filename = output_dir_obj / f"frame_{i:06d}_time_{time_in_seconds:.3f}s.png"
            try:
                video_clip.save_frame(str(frame_filename), t=time_in_seconds)
                extracted_frames_paths.append(str(frame_filename))
                if i % (frame_rate * 10) == 0: # Log progress every 10 seconds of video
                     logger.info(f"Extracted frame for time {time_in_seconds:.2f}s from {video_path_obj.name}")
            except Exception as frame_save_e:
                logger.warning(f"Could not save frame at time {time_in_seconds:.2f}s for {video_path_obj.name}: {frame_save_e}")


        video_clip.close()
        logger.info(f"Successfully extracted {len(extracted_frames_paths)} frames to {output_dir} for {video_path_obj.name}")
        return extracted_frames_paths
    except Exception as e:
        logger.error(f"Error during frame extraction for {video_path}: {e}", exc_info=True)
        raise