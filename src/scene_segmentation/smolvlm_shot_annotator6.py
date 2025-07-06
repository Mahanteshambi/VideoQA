import cv2
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import logging
import uuid
import moviepy.editor as mp
import traceback
import json
import os
from pathlib import Path
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector

# Configure environment for optimal performance
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)

class SmolVLMAnnotator:
    """
    SmolVLM annotator using transformers for reliable inference on EC2 GPU instances.
    """

    def __init__(self, model_id: str = "HuggingFaceTB/SmolVLM-Instruct") -> None:
        logger.info(f"Initializing SmolVLM model: {model_id} for EC2 GPU inference...")
        
        # Set device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load processor and model
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        ).to(self.device)
        
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
        start_time = shot_info["start_seconds"]
        end_time = shot_info["end_seconds"]

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
        Extract metadata for a shot using transformers for reliable inference.
        """
        logger.info(f"Processing shot {shot_info['shot_number']} with SmolVLM...")
        
        # Extract a representative frame from the shot
        frame = self._extract_frame_from_shot(original_video_path, shot_info)
        if frame is None:
            return {"error": "Failed to extract frame from shot"}

        try:
            # Create input messages for SmolVLM
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": self.UNIFIED_JSON_EXTRACTION_PROMPT}
                    ]
                }
            ]
            
            # Prepare inputs using processor
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.processor(text=prompt, images=[frame], return_tensors="pt")
            inputs = inputs.to(self.device)
            
            # Generate response
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.1,
                do_sample=True,
                top_p=0.9
            )
            
            # Decode the response
            generated_texts = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
            )
            
            if generated_texts and len(generated_texts) > 0:
                response_text = generated_texts[0].strip()
                
                # Try to parse JSON from the response
                try:
                    # Extract JSON from the response (it might have extra text)
                    json_start = response_text.find('{')
                    json_end = response_text.rfind('}') + 1
                    
                    if json_start != -1 and json_end > json_start:
                        json_str = response_text[json_start:json_end]
                        response = json.loads(json_str)
                    else:
                        # If no JSON found, return the raw text with a wrapper
                        response = {"ShotDescription": response_text, "raw_response": response_text}
                        
                except json.JSONDecodeError:
                    # If JSON parsing fails, return the raw text with error info
                    response = {"ShotDescription": response_text, "parsing_error": True, "raw_response": response_text}
                
                logger.info(f"Successfully processed shot {shot_info['shot_number']}")
                return response
            else:
                logger.error(f"No output generated for shot {shot_info['shot_number']}")
                return {"error": "No output generated"}

        except Exception as e:
            logger.error(f"Error processing shot {shot_info['shot_number']}: {e}")
            return {"error": f"Processing error: {str(e)}"}
    
    def _extract_frame_from_shot(self, video_path: str, shot_info: dict) -> Image.Image | None:
        """
        Extract a representative frame from a shot for analysis.
        """
        try:
            with mp.VideoFileClip(video_path) as video:
                start_time = shot_info["start_seconds"]
                end_time = shot_info["end_seconds"]
                
                # Extract middle frame
                middle_time = start_time + (end_time - start_time) / 2
                
                if middle_time < video.duration:
                    frame = video.get_frame(middle_time)
                    # Convert numpy array to PIL Image
                    pil_image = Image.fromarray(frame)
                    return pil_image
                else:
                    return None
        except Exception as e:
            logger.error(f"Error extracting frame: {e}")
            return None

    def process_video_shots(self, video_path: str, output_file: str = None) -> list:
        """
        Process all shots in a video using transformers for reliable processing.
        """
        logger.info(f"Starting SmolVLM-based shot processing for: {video_path}")
        
        # Detect shots
        shots = self._detect_video_shots(video_path)
        logger.info(f"Detected {len(shots)} shots in video")
        
        results = []
        
        # Process shots with SmolVLM
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
    output_json_file = "hair_love_smolvlm_metadata.json"

    # --- Initialize SmolVLM annotator ---
    annotator = SmolVLMAnnotator()
    
    # --- Process all shots ---
    results = annotator.process_video_shots(video_path, output_json_file)
    
    logger.info(f"\n✅ SmolVLM processing complete. Processed {len(results)} shots.")
    logger.info(f"Results saved to: {output_json_file}")
    
    # Print summary
    print(f"\n=== SmolVLM Processing Summary ===")
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