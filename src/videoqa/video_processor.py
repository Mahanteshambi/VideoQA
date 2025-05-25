# video_processor.py

import os
import sys
import json
from pathlib import Path
import shutil  # For checking ffmpeg installation

# Ensure ffmpeg is in the PATH
os.environ["PATH"] = "/opt/homebrew/bin:" + os.environ.get("PATH", "")

# Now import ffmpeg-related modules
import ffmpeg
import moviepy.editor as mp
import whisper

def check_ffmpeg_availability():
    """
    Checks if the ffmpeg command is available on the system PATH.
    Raises FileNotFoundError if ffmpeg is not found.
    """
    ffmpeg_path = shutil.which("ffmpeg")
    ffprobe_path = shutil.which("ffprobe")
    
    if ffmpeg_path is None:
        raise FileNotFoundError(
            f"FFmpeg not found in PATH: {os.environ['PATH']}. Please install FFmpeg and ensure it is in your system's PATH. "
            "Download from https://ffmpeg.org/download.html"
        )
    if ffprobe_path is None:
        raise FileNotFoundError(
            f"ffprobe not found in PATH: {os.environ['PATH']}. Please ensure ffprobe (part of FFmpeg) is in your system's PATH."
        )
    print(f"FFmpeg found at: {ffmpeg_path}")
    print(f"ffprobe found at: {ffprobe_path}")

def extract_frames(video_path: str, output_dir: str, frame_rate: int = 1) -> list[str]:
    """
    Extracts frames from a video file using MoviePy.

    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Directory to save the extracted frames.
        frame_rate (int): Number of frames to extract per second.

    Returns:
        list[str]: A list of paths to the extracted frame images.
    
    Raises:
        FileNotFoundError: If the video_path does not exist.
        Exception: If frame extraction fails.
    """
    video_path_obj = Path(video_path)
    if not video_path_obj.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    output_dir_obj = Path(output_dir)
    output_dir_obj.mkdir(parents=True, exist_ok=True)

    extracted_frames_paths = []
    try:
        print(f"Starting frame extraction for {video_path_obj.name} at {frame_rate} fps...")
        video_clip = mp.VideoFileClip(video_path)
        duration = video_clip.duration
        
        for i, frame_time in enumerate(range(0, int(duration * frame_rate))):
            time_in_seconds = frame_time / frame_rate
            frame_filename = output_dir_obj / f"frame_{i:05d}_time_{time_in_seconds:.2f}s.png"
            video_clip.save_frame(str(frame_filename), t=time_in_seconds)
            extracted_frames_paths.append(str(frame_filename))
            if i % (frame_rate * 10) == 0: # Log progress every 10 seconds of video
                 print(f"Extracted frame for time {time_in_seconds:.2f}s")

        video_clip.close() # Release resources
        print(f"Successfully extracted {len(extracted_frames_paths)} frames to {output_dir}")
        return extracted_frames_paths
    except Exception as e:
        print(f"Error during frame extraction for {video_path}: {e}")
        # Clean up partially extracted frames if necessary
        # for p in extracted_frames_paths:
        #     if Path(p).exists(): Path(p).unlink()
        # if output_dir_obj.exists() and not any(output_dir_obj.iterdir()):
        #     output_dir_obj.rmdir()
        raise

def extract_audio(video_path: str, output_audio_path: str) -> str:
    """
    Extracts the audio track from a video file using ffmpeg-python.

    Args:
        video_path (str): Path to the input video file.
        output_audio_path (str): Path to save the extracted audio file (e.g., 'audio.mp3', 'audio.wav').

    Returns:
        str: The path to the extracted audio file.

    Raises:
        FileNotFoundError: If the video_path does not exist.
        ffmpeg.Error: If FFmpeg encounters an error during audio extraction.
    """
    video_path_obj = Path(video_path)
    if not video_path_obj.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    output_audio_path_obj = Path(output_audio_path)
    output_audio_path_obj.parent.mkdir(parents=True, exist_ok=True)

    try:
        print(f"Starting audio extraction for {video_path_obj.name}...")
        # Use ffmpeg-python to extract audio. AAC is a common codec for MP3/MP4 audio.
        # For WAV, use pcm_s16le or similar.
        audio_codec = 'aac' if output_audio_path_obj.suffix.lower() == '.mp3' else 'pcm_s16le' # Default to WAV if not mp3

        (
            ffmpeg
            .input(video_path)
            .output(output_audio_path, acodec=audio_codec, vn=None) # vn=None means no video
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        print(f"Successfully extracted audio to {output_audio_path}")
        return output_audio_path
    except ffmpeg.Error as e:
        print(f"Error during audio extraction for {video_path}:")
        print(f"FFmpeg stdout: {e.stdout.decode('utf8')}")
        print(f"FFmpeg stderr: {e.stderr.decode('utf8')}")
        raise
    except Exception as e_gen:
        print(f"An unexpected error occurred during audio extraction: {e_gen}")
        raise


def transcribe_audio(audio_path: str, output_transcript_dir: str, model_name: str = "base") -> dict:
    """
    Transcribes an audio file using OpenAI Whisper.

    Args:
        audio_path (str): Path to the input audio file.
        output_transcript_dir (str): Directory to save the transcription result (JSON and TXT).
        model_name (str): Name of the Whisper model to use (e.g., "tiny", "base", "small").

    Returns:
        dict: The transcription result from Whisper.
    
    Raises:
        FileNotFoundError: If the audio_path does not exist.
        Exception: If transcription fails.
    """
    audio_path_obj = Path(audio_path)
    if not audio_path_obj.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    output_dir_obj = Path(output_transcript_dir)
    output_dir_obj.mkdir(parents=True, exist_ok=True)
    
    transcript_json_path = output_dir_obj / f"{audio_path_obj.stem}_transcription.json"
    transcript_txt_path = output_dir_obj / f"{audio_path_obj.stem}_transcription.txt"

    try:
        print(f"Loading Whisper model '{model_name}'...")
        # Note: Whisper can auto-detect if a GPU is available.
        # Forcing CPU: model = whisper.load_model(model_name, device="cpu")
        model = whisper.load_model(model_name)
        print(f"Starting audio transcription for {audio_path_obj.name} (this may take a while)...")
        
        result = model.transcribe(audio_path, verbose=True) # verbose=True shows progress
        
        # Save the full result as JSON
        with open(transcript_json_path, "w", encoding="utf-8") as f_json:
            json.dump(result, f_json, indent=4, ensure_ascii=False)
        
        # Save just the text as TXT
        with open(transcript_txt_path, "w", encoding="utf-8") as f_txt:
            f_txt.write(result["text"])
            
        print(f"Successfully transcribed audio. JSON saved to {transcript_json_path}, TXT saved to {transcript_txt_path}")
        return result
    except Exception as e:
        print(f"Error during audio transcription for {audio_path}: {e}")
        raise

def get_video_metadata(video_path: str) -> dict:
    """
    Extracts basic technical metadata from a video file using ffprobe.

    Args:
        video_path (str): Path to the input video file.

    Returns:
        dict: A dictionary containing video metadata.
              Returns an empty dict if ffprobe fails or video has no streams.
    
    Raises:
        FileNotFoundError: If the video_path does not exist.
        ffmpeg.Error: If ffprobe encounters an error.
    """
    video_path_obj = Path(video_path)
    if not video_path_obj.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    try:
        print(f"Extracting metadata for {video_path_obj.name}...")
        probe = ffmpeg.probe(video_path)
        
        video_streams = [s for s in probe.get('streams', []) if s.get('codec_type') == 'video']
        audio_streams = [s for s in probe.get('streams', []) if s.get('codec_type') == 'audio']
        
        metadata = {
            "filename": video_path_obj.name,
            "format_name": probe.get('format', {}).get('format_name'),
            "duration_seconds": float(probe.get('format', {}).get('duration', 0)),
            "size_bytes": int(probe.get('format', {}).get('size', 0)),
            "bit_rate_bps": int(probe.get('format', {}).get('bit_rate', 0)),
        }
        
        if video_streams:
            vs = video_streams[0] # Taking the first video stream
            metadata["video_codec"] = vs.get('codec_name')
            metadata["width"] = vs.get('width')
            metadata["height"] = vs.get('height')
            metadata["fps"] = eval(vs.get('r_frame_rate', '0/1')) # e.g., "30000/1001"
            metadata["video_bit_rate_bps"] = int(vs.get('bit_rate', 0)) if vs.get('bit_rate') else None

        if audio_streams:
            auds = audio_streams[0] # Taking the first audio stream
            metadata["audio_codec"] = auds.get('codec_name')
            metadata["sample_rate_hz"] = int(auds.get('sample_rate', 0))
            metadata["channels"] = auds.get('channels')
            metadata["audio_bit_rate_bps"] = int(auds.get('bit_rate',0)) if auds.get('bit_rate') else None
            
        print(f"Successfully extracted metadata for {video_path_obj.name}")
        return metadata
    except ffmpeg.Error as e:
        print(f"Error probing video {video_path}:")
        print(f"FFprobe stdout: {e.stdout.decode('utf8')}")
        print(f"FFprobe stderr: {e.stderr.decode('utf8')}")
        raise
    except Exception as e_gen:
        print(f"An unexpected error occurred during metadata extraction: {e_gen}")
        raise


def process_video(
    video_path: str, 
    base_output_dir: str, 
    frame_rate: int = 1, 
    whisper_model: str = "base",
    skip_frames: bool = False,
    skip_audio: bool = False,
    skip_transcription: bool = False,
    skip_metadata: bool = False
) -> dict:
    """
    Main orchestrator function to process a single video file.
    It extracts frames, audio, transcribes audio, and collects metadata.

    Args:
        video_path (str): Path to the input video file.
        base_output_dir (str): The base directory where all processed outputs will be stored.
                               A subdirectory named after the video file will be created here.
        frame_rate (int): Frames per second to extract.
        whisper_model (str): Whisper model name for transcription.
        skip_frames (bool): If True, skips frame extraction.
        skip_audio (bool): If True, skips audio extraction (and consequently transcription).
        skip_transcription (bool): If True, skips audio transcription.
        skip_metadata (bool): If True, skips metadata extraction.

    Returns:
        dict: A dictionary containing paths to the generated artifacts and metadata.
              Example: 
              {
                  "original_video_path": "path/to/video.mp4",
                  "processed_video_dir": "base_output/video/",
                  "frames_dir": "base_output/video/frames/",
                  "extracted_frames_paths": ["path/to/frame1.png", ...],
                  "audio_file_path": "base_output/video/audio/audio.mp3",
                  "transcript_dir": "base_output/video/transcripts/",
                  "transcription_json_path": "...",
                  "transcription_txt_path": "...",
                  "metadata": {...}
              }
    """
    try:
        check_ffmpeg_availability() # Check once at the beginning
    except FileNotFoundError as e:
        print(f"Critical error: {e}")
        return {"error": str(e)}

    video_path_obj = Path(video_path)
    if not video_path_obj.is_file():
        print(f"Error: Video file {video_path} not found or is not a file.")
        return {"error": f"Video file {video_path} not found."}

    video_name_stem = video_path_obj.stem
    processed_video_dir = Path(base_output_dir) / video_name_stem
    processed_video_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "original_video_path": str(video_path_obj),
        "processed_video_dir": str(processed_video_dir),
        "frames_dir": None,
        "extracted_frames_paths": [],
        "audio_file_path": None,
        "transcript_dir": None,
        "transcription_json_path": None,
        "transcription_txt_path": None,
        "metadata": None,
        "errors": []
    }

    # 1. Extract Metadata
    if not skip_metadata:
        try:
            print(f"\n--- Step 1: Extracting Metadata for {video_name_stem} ---")
            results["metadata"] = get_video_metadata(video_path)
        except Exception as e:
            err_msg = f"Failed metadata extraction: {e}"
            print(err_msg)
            results["errors"].append(err_msg)
    else:
        print(f"\n--- Step 1: Skipping Metadata Extraction for {video_name_stem} ---")


    # 2. Extract Frames
    if not skip_frames:
        frames_output_dir = processed_video_dir / "frames"
        results["frames_dir"] = str(frames_output_dir)
        try:
            print(f"\n--- Step 2: Extracting Frames for {video_name_stem} ---")
            results["extracted_frames_paths"] = extract_frames(video_path, str(frames_output_dir), frame_rate)
        except Exception as e:
            err_msg = f"Failed frame extraction: {e}"
            print(err_msg)
            results["errors"].append(err_msg)
    else:
        print(f"\n--- Step 2: Skipping Frame Extraction for {video_name_stem} ---")

    # 3. Extract Audio
    audio_file_path = None
    if not skip_audio:
        audio_output_dir = processed_video_dir / "audio"
        audio_output_dir.mkdir(parents=True, exist_ok=True)
        # Using .wav for transcription as it's lossless and Whisper handles it well.
        # MP3 can also work but might involve lossy compression.
        audio_file_path = audio_output_dir / f"{video_name_stem}_audio.wav" 
        results["audio_file_path"] = str(audio_file_path)
        try:
            print(f"\n--- Step 3: Extracting Audio for {video_name_stem} ---")
            extract_audio(video_path, str(audio_file_path))
        except Exception as e:
            err_msg = f"Failed audio extraction: {e}"
            print(err_msg)
            results["errors"].append(err_msg)
            audio_file_path = None # Ensure it's None if extraction failed
            results["audio_file_path"] = None
    else:
        print(f"\n--- Step 3: Skipping Audio Extraction for {video_name_stem} ---")
        if not skip_transcription: # If skipping audio, must skip transcription
            print("Skipping transcription as audio extraction is skipped.")
            skip_transcription = True


    # 4. Transcribe Audio
    if not skip_transcription and audio_file_path and Path(audio_file_path).exists():
        transcript_output_dir = processed_video_dir / "transcripts"
        results["transcript_dir"] = str(transcript_output_dir)
        try:
            print(f"\n--- Step 4: Transcribing Audio for {video_name_stem} ---")
            transcribe_audio(str(audio_file_path), str(transcript_output_dir), whisper_model)
            results["transcription_json_path"] = str(transcript_output_dir / f"{Path(audio_file_path).stem}_transcription.json")
            results["transcription_txt_path"] = str(transcript_output_dir / f"{Path(audio_file_path).stem}_transcription.txt")
        except Exception as e:
            err_msg = f"Failed audio transcription: {e}"
            print(err_msg)
            results["errors"].append(err_msg)
    elif not skip_transcription and not audio_file_path:
         err_msg = "Skipping transcription because audio extraction failed or was skipped."
         print(err_msg)
         results["errors"].append(err_msg)
    else:
        print(f"\n--- Step 4: Skipping Audio Transcription for {video_name_stem} ---")
        
    print(f"\n--- Processing complete for {video_name_stem} ---")
    if results["errors"]:
        print("Completed with errors:")
        for err in results["errors"]:
            print(f"- {err}")
    else:
        print("Completed successfully.")
        
    return results

if __name__ == "__main__":
    print("Starting video processing module demo...")
    
    # --- Configuration for Demo ---
    # Create a dummy video file for testing if it doesn't exist
    # In a real scenario, you would point this to an actual video file.
    sample_video_path = "Hair Love.webm" 
    base_output_directory = "processed_videos_output"

    # Check if a sample video exists, if not, try to create a tiny one with ffmpeg (if available)
    # This is just for making the example runnable.
    # For actual use, replace sample_video.mp4 with your video.
    if not Path(sample_video_path).exists():
        print(f"Sample video '{sample_video_path}' not found.")
        print("Attempting to create a short dummy MP4 video for testing using FFmpeg.")
        print("If this fails, please create a sample_video.mp4 manually or provide a path to an existing video.")
        try:
            check_ffmpeg_availability() # Check if ffmpeg is available for creating dummy video
            (
                ffmpeg
                .input('color=c=blue:s=128x128:d=2', format='lavfi') # 2 second blue video
                .output(sample_video_path, vcodec='libx264', pix_fmt='yuv420p', acodec='aac', ar='44100', t=2) # Ensure audio stream
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
            print(f"Created dummy '{sample_video_path}'.")
        except Exception as e:
            print(f"Could not create dummy video: {e}")
            print("Please ensure 'sample_video.mp4' exists or provide a path to your video.")
            exit()
    
    # --- Run the processing ---
    if Path(sample_video_path).exists():
        processing_results = process_video(
            video_path=sample_video_path,
            base_output_dir=base_output_directory,
            frame_rate=1, # Extract 1 frame per second
            whisper_model="tiny", # Use "tiny" for faster demo, "base" or "small" for better accuracy
            # skip_frames=True,
            # skip_audio=True,
            # skip_transcription=True,
            # skip_metadata=True
        )
        print("\n--- Final Processing Results ---")
        print(json.dumps(processing_results, indent=4))
    else:
        print(f"Cannot run demo: '{sample_video_path}' still not found.")
