# video_analysis_project/src/video_ingestion/audio_processing.py

import ffmpeg
import whisper
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def extract_audio(video_path: str, output_audio_path: str) -> str | None:
    """
    Extracts the audio track from a video file using ffmpeg-python.
    (Content is similar to the previous version)
    """
    video_path_obj = Path(video_path)
    if not video_path_obj.exists():
        logger.error(f"Video file not found: {video_path}")
        raise FileNotFoundError(f"Video file not found: {video_path}")

    output_audio_path_obj = Path(output_audio_path)
    output_audio_path_obj.parent.mkdir(parents=True, exist_ok=True)

    try:
        logger.info(f"Starting audio extraction for {video_path_obj.name} to {output_audio_path_obj.name}...")
        audio_codec = 'aac' if output_audio_path_obj.suffix.lower() == '.mp3' else 'pcm_s16le'
        bitrate = "192k" if output_audio_path_obj.suffix.lower() == '.mp3' else None


        stream = ffmpeg.input(video_path)
        if bitrate:
             stream = ffmpeg.output(stream, output_audio_path, acodec=audio_codec, audio_bitrate=bitrate, vn=None)
        else:
             stream = ffmpeg.output(stream, output_audio_path, acodec=audio_codec, vn=None)
        
        stream.overwrite_output().run(capture_stdout=True, capture_stderr=True, quiet=True)
        
        logger.info(f"Successfully extracted audio to {output_audio_path}")
        return output_audio_path
    except ffmpeg.Error as e:
        stderr = e.stderr.decode('utf8') if e.stderr else "No stderr"
        logger.error(f"Error during audio extraction for {video_path}: {stderr}", exc_info=True)
        # Attempt to delete partially created file if it's empty or invalid
        if output_audio_path_obj.exists() and output_audio_path_obj.stat().st_size == 0:
            try:
                output_audio_path_obj.unlink()
                logger.info(f"Deleted empty/partial audio file: {output_audio_path}")
            except OSError as ose:
                logger.warning(f"Could not delete partial audio file {output_audio_path}: {ose}")
        return None # Indicate failure
    except Exception as e_gen:
        logger.error(f"An unexpected error occurred during audio extraction for {video_path}: {e_gen}", exc_info=True)
        return None


def transcribe_audio(audio_path: str, output_transcript_dir: str, model_name: str = "base") -> tuple[str | None, str | None]:
    """
    Transcribes an audio file using OpenAI Whisper.
    Returns paths to JSON and TXT transcriptions or None if failed.
    (Content is similar to the previous version, added better return for paths)
    """
    audio_path_obj = Path(audio_path)
    if not audio_path_obj.exists() or audio_path_obj.stat().st_size == 0:
        logger.error(f"Audio file not found or is empty: {audio_path}")
        return None, None

    output_dir_obj = Path(output_transcript_dir)
    output_dir_obj.mkdir(parents=True, exist_ok=True)
    
    transcript_base_name = f"{audio_path_obj.stem}_transcription"
    transcript_json_path = output_dir_obj / f"{transcript_base_name}.json"
    transcript_txt_path = output_dir_obj / f"{transcript_base_name}.txt"

    try:
        logger.info(f"Loading Whisper model '{model_name}'...")
        model = whisper.load_model(model_name)
        logger.info(f"Starting audio transcription for {audio_path_obj.name} (model: {model_name})...")
        
        # Set verbose=False for less console output during transcription, use logger for progress if needed
        result = model.transcribe(str(audio_path_obj), verbose=False) 
        
        with open(transcript_json_path, "w", encoding="utf-8") as f_json:
            json.dump(result, f_json, indent=2, ensure_ascii=False)
        
        with open(transcript_txt_path, "w", encoding="utf-8") as f_txt:
            f_txt.write(result["text"])
            
        logger.info(f"Successfully transcribed audio. JSON: {transcript_json_path}, TXT: {transcript_txt_path}")
        return str(transcript_json_path), str(transcript_txt_path)
    except Exception as e:
        logger.error(f"Error during audio transcription for {audio_path}: {e}", exc_info=True)
        return None, None