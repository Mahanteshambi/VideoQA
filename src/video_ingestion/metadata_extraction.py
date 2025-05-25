# video_analysis_project/src/video_ingestion/metadata_extraction.py

import ffmpeg
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)

def get_video_metadata(video_path: str) -> dict | None:
    """
    Extracts basic technical metadata from a video file using ffprobe.
    (Content is similar to the previous version)
    """
    video_path_obj = Path(video_path)
    if not video_path_obj.exists():
        logger.error(f"Video file not found: {video_path}")
        raise FileNotFoundError(f"Video file not found: {video_path}")

    try:
        logger.info(f"Extracting metadata for {video_path_obj.name}...")
        probe = ffmpeg.probe(str(video_path))
        
        video_streams = [s for s in probe.get('streams', []) if s.get('codec_type') == 'video']
        audio_streams = [s for s in probe.get('streams', []) if s.get('codec_type') == 'audio']
        
        metadata = {
            "filename": video_path_obj.name,
            "filepath": str(video_path_obj.resolve()),
            "format_name": probe.get('format', {}).get('format_name'),
            "duration_seconds": float(probe.get('format', {}).get('duration', 0.0)),
            "size_bytes": int(probe.get('format', {}).get('size', 0)),
            "bit_rate_bps": int(probe.get('format', {}).get('bit_rate', 0)),
            "creation_time": probe.get('format', {}).get('tags', {}).get('creation_time')
        }
        
        if video_streams:
            vs = video_streams[0]
            metadata["video_codec"] = vs.get('codec_name')
            metadata["width"] = int(vs.get('width',0))
            metadata["height"] = int(vs.get('height',0))
            try:
                if 'r_frame_rate' in vs and vs['r_frame_rate'] and '/' in vs['r_frame_rate']:
                    num, den = map(int, vs['r_frame_rate'].split('/'))
                    metadata["fps"] = float(num / den) if den != 0 else 0.0
                else:
                    metadata["fps"] = 0.0
            except (ValueError, TypeError, ZeroDivisionError):
                 metadata["fps"] = 0.0 # Default if parsing fails
            metadata["video_bit_rate_bps"] = int(vs.get('bit_rate', 0)) if vs.get('bit_rate') else None
            metadata["pix_fmt"] = vs.get('pix_fmt')


        if audio_streams:
            auds = audio_streams[0]
            metadata["audio_codec"] = auds.get('codec_name')
            metadata["sample_rate_hz"] = int(auds.get('sample_rate', 0))
            metadata["channels"] = int(auds.get('channels',0))
            metadata["audio_bit_rate_bps"] = int(auds.get('bit_rate',0)) if auds.get('bit_rate') else None
            
        logger.info(f"Successfully extracted metadata for {video_path_obj.name}")
        return metadata
    except ffmpeg.Error as e:
        stderr = e.stderr.decode('utf8') if e.stderr else "No stderr"
        logger.error(f"Error probing video {video_path}: {stderr}", exc_info=True)
        return None
    except Exception as e_gen:
        logger.error(f"An unexpected error occurred during metadata extraction for {video_path}: {e_gen}", exc_info=True)
        return None