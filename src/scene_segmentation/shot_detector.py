# video_analysis_project/src/scene_segmentation/shot_detector.py

from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector # Or ThresholdDetector
from scenedetect.scene_manager import save_images # For keyframes if needed later
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def detect_shots_pyscenedetect(video_path: str, threshold: float = 27.0, min_scene_len_frames: int = 15) -> list[dict]:
    """
    Detects shots in a video using PySceneDetect's ContentDetector.

    Args:
        video_path (str): Path to the video file.
        threshold (float): Threshold for the ContentDetector. Lower values detect more scenes/shots.
        min_scene_len_frames (int): Minimum length of a shot in frames.

    Returns:
        list[dict]: A list of shots, where each shot is a dictionary with
                    'shot_number', 'start_time_seconds', 'end_time_seconds',
                    'start_frame', 'end_frame', 'duration_seconds'.
    """
    video_path_obj = Path(video_path)
    if not video_path_obj.exists():
        logger.error(f"Video file not found for shot detection: {video_path}")
        raise FileNotFoundError(f"Video file not found: {video_path}")

    logger.info(f"Starting shot detection for {video_path_obj.name} using PySceneDetect (ContentDetector threshold={threshold})...")
    
    try:
        # Open video and create scene manager
        video = open_video(video_path)
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=threshold, min_scene_len=min_scene_len_frames))
        
        # Perform shot detection
        scene_manager.detect_scenes(video=video, show_progress=True)
        
        # Get the list of shots
        shot_list_raw = scene_manager.get_scene_list()

        shots = []
        if not shot_list_raw:
            logger.warning(f"No distinct shots detected in {video_path_obj.name}. Treating as a single shot.")
            # Get video stats from the video object
            stats = scene_manager.stats_manager
            if stats:
                duration_frames = stats.get_frames()
                fps = stats.get_framerate()
                if duration_frames and fps:
                    duration_seconds = duration_frames / fps
                    shots.append({
                        "shot_number": 1,
                        "start_time_seconds": 0,
                        "end_time_seconds": duration_seconds,
                        "start_frame": 0,
                        "end_frame": duration_frames,
                        "duration_seconds": duration_seconds
                    })
        else:
            for i, (start_tc, end_tc) in enumerate(shot_list_raw):
                shots.append({
                    "shot_number": i + 1,
                    "start_time_seconds": start_tc.get_seconds(),
                    "end_time_seconds": end_tc.get_seconds(),
                    "start_frame": start_tc.get_frames(),
                    "end_frame": end_tc.get_frames(),
                    "duration_seconds": end_tc.get_seconds() - start_tc.get_seconds()
                })
        
        logger.info(f"Detected {len(shots)} shots in {video_path_obj.name}.")
        return shots

    except Exception as e:
        logger.error(f"Error during shot detection for {video_path}: {e}", exc_info=True)
        raise