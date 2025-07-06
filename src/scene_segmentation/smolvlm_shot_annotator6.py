import cv2
import torch
from vllm import LLM, SamplingParams
from vllm.model_executor.models.llava import LlavaForCausalLM
from PIL import Image
import logging
import uuid
import moviepy.editor as mp
import traceback
import json
from pathlib import Path
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector

logger = logging.getLogger(__name__)

class VLLMSmolVLMAnnotator:
    """
    VLLM-optimized SmolVLM annotator for fast inference on EC2 GPU instances.
    """

    def __init__(self, model_id: str = "HuggingFaceTB/SmolVLM2-2.2B-Instruct") -> None:
        logger.info(f"Initializing VLLM model: {model_id} for EC2 GPU inference...")
        
        # VLLM configuration for optimal performance on EC2
        self.llm = LLM(
            model=model_id,
            trust_remote_code=True,
            dtype="bfloat16",  # Use bfloat16 for better performance on modern GPUs
            gpu_memory_utilization=0.9,  # Use 90% of GPU memory
            max_model_len=4096,  # Adjust based on your GPU memory
            enforce_eager=True,  # Better for smaller models
            tensor_parallel_size=1,  # Single GPU setup
        )
        
        self.UNIFIED_JSON_EXTRACTION_PROMPT = """You are a scene understanding assistant. Analyze the provided video shot carefully. Extract and return metadata solely based on the visual and implied content of this shot.

            Respond with one valid, complete, and properly formatted JSON object that strictly follows the schema below. Do not include any explanations or comments. Only output the filled JSON object.

            ⛔ Do not repeat or describe the schema itself.
            ✅ Do include specific values based on your analysis of the video content.

            Ensure your JSON response:
            - Has all required keys as per schema (do not rename fields)
            - Uses properly quoted strings and comma-separated lists
            - Closes all arrays and objects correctly
            - Fills all values realistically based on the video

            🎯 Fill the following schema:

            {
            "ShotDescription": "...",
            "GenreCues": [
                {
                "genre_hint": "...",
                "prominence_in_shot": ...
                }
            ],
            "SubgenreCues": ["..."],
            "AdjectiveTheme": ["..."],
            "Mood": ["..."],
            "SettingContext": ["..."],
            "ContentDescriptors": ["..."],
            "LocationHints_Regional": ["..."],
            "LocationHints_International": ["..."],
            "SearchKeywords": ["..."]
            }
"""

    def _create_temp_shot_video(self, original_video_path: str, shot_info: dict) -> str | None:
        """
        Extracts a single shot from the main video and saves it as a temporary video file.
        Returns the path to the temporary file.
        """
        start_time = shot_info["start_time_seconds"]
        end_time = shot_info["end_time_seconds"]

        if end_time <= start_time:
            logger.warning(f"Shot {shot_info.get('shot_number')} has no duration. Skipping.")
            return None
            
        temp_dir = Path("/tmp/vllm_smolvlm_shots")
        temp_dir.mkdir(exist_ok=True)
        temp_video_path = str(temp_dir / f"shot_{shot_info['shot_number']}_{uuid.uuid4()}.mp4")
        
        try:
            # Use moviepy to create a subclip of the shot
            with mp.VideoFileClip(original_video_path) as video:
                video.subclip(start_time, end_time).write_videofile(
                    temp_video_path, 
                    codec="libx264", 
                    audio_codec="aac",
                    verbose=False, 
                    logger=None
                )
            return temp_video_path
        except Exception as e:
            logger.error(f"Failed to create temporary video for shot {shot_info['shot_number']}: {e}\n{traceback.format_exc()}")
            return None

    def extract_metadata_for_shot(self, original_video_path: str, shot_info: dict) -> dict:
        """
        Extract metadata for a shot using VLLM for fast inference.
        """
        logger.info(f"Processing shot {shot_info['shot_number']} with VLLM...")
        
        # Create temporary video for the shot
        temp_video_path = self._create_temp_shot_video(original_video_path, shot_info)
        if not temp_video_path:
            return {"error": "Failed to create temporary video for shot"}

        try:
            # Prepare the prompt with video
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "path": temp_video_path},
                        {"type": "text", "text": self.UNIFIED_JSON_EXTRACTION_PROMPT}
                    ]
                }
            ]

            # VLLM sampling parameters for optimal performance
            sampling_params = SamplingParams(
                temperature=0.1,  # Low temperature for consistent JSON output
                top_p=0.9,
                max_tokens=1024,
                stop=["</s>", "<|endoftext|>", "Human:", "Assistant:"]
            )

            # Generate with VLLM (much faster than standard transformers)
            outputs = self.llm.generate(messages, sampling_params)
            
            if outputs and len(outputs) > 0:
                response = outputs[0].outputs[0].text.strip()
                
                # Clean up temporary file
                try:
                    Path(temp_video_path).unlink()
                except:
                    pass
                
                logger.info(f"Successfully processed shot {shot_info['shot_number']}")
                return response
            else:
                logger.error(f"No output generated for shot {shot_info['shot_number']}")
                return {"error": "No output generated"}

        except Exception as e:
            logger.error(f"Error processing shot {shot_info['shot_number']}: {e}")
            return {"error": f"Processing error: {str(e)}"}

    def process_video_shots(self, video_path: str, output_file: str = None) -> list:
        """
        Process all shots in a video using VLLM for batch processing.
        """
        logger.info(f"Starting VLLM-based shot processing for: {video_path}")
        
        # Detect shots
        shots = self._detect_video_shots(video_path)
        logger.info(f"Detected {len(shots)} shots in video")
        
        results = []
        
        # Process shots with VLLM (can be batched for even better performance)
        for shot in shots:
            logger.info(f"Processing shot {shot['shot_number']}/{len(shots)}")
            
            metadata = self.extract_metadata_for_shot(video_path, shot)
            
            # Add shot info to metadata
            shot_result = {
                "shot_number": shot["shot_number"],
                "start_time_seconds": shot["start_seconds"],
                "end_time_seconds": shot["end_seconds"],
                "metadata": metadata
            }
            
            results.append(shot_result)
            
            # Print progress
            print(f"Shot {shot['shot_number']}: {metadata.get('ShotDescription', 'Processing...')[:100]}...")
        
        # Save results
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to: {output_file}")
        
        return results

    def _detect_video_shots(self, video_path: str) -> list:
        """Detect shots in video using PySceneDetect."""
        video = open_video(video_path)
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=27))
        scene_manager.detect_scenes(video, show_progress=True)
        shot_list = scene_manager.get_scene_list()
        
        return [{
            "shot_number": i + 1,
            "start_seconds": shot[0].get_seconds(),
            "end_seconds": shot[1].get_seconds()
        } for i, shot in enumerate(shot_list)]


def find_video_shots(video_path: str) -> list:
    """Uses PySceneDetect to find all shots in a video."""
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=27))
    scene_manager.detect_scenes(video, show_progress=True)
    shot_list = scene_manager.get_scene_list()
    return [{
        "shot_number": i + 1,
        "start_seconds": shot[0].get_seconds(),
        "end_seconds": shot[1].get_seconds()
    } for i, shot in enumerate(shot_list)]


def extract_keyframe_as_image(video_capture: cv2.VideoCapture, shot_info: dict) -> Image.Image | None:
    """Extracts the middle frame of a shot and returns it as a PIL Image."""
    middle_timestamp_msec = (shot_info["start_seconds"] + (shot_info["end_seconds"] - shot_info["start_seconds"]) / 2) * 1000
    video_capture.set(cv2.CAP_PROP_POS_MSEC, middle_timestamp_msec)
    success, frame = video_capture.read()
    if success:
        # Convert from OpenCV's BGR format to RGB for PIL
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return None


if __name__ == "__main__":
    # --- Configuration for EC2 ---
    video_path = "sample_videos/Hair Love.mp4"
    output_json_file = "hair_love_vllm_metadata.json"

    # --- Initialize VLLM annotator ---
    annotator = VLLMSmolVLMAnnotator()
    
    # --- Process all shots ---
    results = annotator.process_video_shots(video_path, output_json_file)
    
    logger.info(f"\n✅ VLLM processing complete. Processed {len(results)} shots.")
    logger.info(f"Results saved to: {output_json_file}")
    
    # Print summary
    print(f"\n=== VLLM Processing Summary ===")
    print(f"Total shots processed: {len(results)}")
    print(f"Output file: {output_json_file}")
    
    # Show first few results
    for i, result in enumerate(results[:3]):
        print(f"\nShot {result['shot_number']}:")
        metadata = result['metadata']
        if isinstance(metadata, dict) and 'ShotDescription' in metadata:
            print(f"  Description: {metadata['ShotDescription'][:100]}...")
        else:
            print(f"  Status: {metadata}")