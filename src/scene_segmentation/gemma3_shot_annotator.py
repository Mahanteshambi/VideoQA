import json
import os
import cv2
import torch
import traceback
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector
import logging

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Gemma3nShotAnnotator:
    def __init__(self, model_path: str = "google/gemma-3n-e4b"): # <-- THE ONLY MAJOR CHANGE
        """
        Initializes the Gemma3n model and processor.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing model {model_path} on device: {self.device}")

        # The class names might differ slightly for a new model,
        # but the pattern will be the same.
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        ).to(self.device)

        self.UNIFIED_JSON_EXTRACTION_PROMPT = """Analyze the provided image, which is a representative frame from a video shot. Your task is to extract comprehensive metadata based SOLELY on the visual and implied content of THIS IMAGE.
            Respond ONLY with a single, valid, properly formatted JSON object. Do not add any explanatory text before or after the JSON.
            Adhere strictly to the following JSON schema:
            {
            "ShotDescription": "string",
            "GenreCues": [{ "genre_hint": "string", "prominence_in_shot": "integer" }],
            "SubgenreCues": ["string"],
            "AdjectiveTheme": ["string"],
            "Mood": ["string"],
            "SettingContext": ["string"],
            "ContentDescriptors": ["string"],
            "LocationHints_Regional": ["string"],
            "LocationHints_International": ["string"],
            "SearchKeywords": ["string"]
            }
            """

    def generate_json_for_frame(self, frame_path: str) -> dict | None:
        """
        Generates a JSON metadata object for a single image frame.
        This is the corrected core function.
        """
        try:
            image = Image.open(frame_path)
        except FileNotFoundError:
            logger.error(f"Frame image not found at {frame_path}")
            return None

        # This is the correct way to process image and text for this model
        inputs = self.processor(
            text=self.UNIFIED_JSON_EXTRACTION_PROMPT,
            images=image,
            return_tensors="pt"
        ).to(self.device)

        # Generate the text output
        generated_ids = self.model.generate(**inputs, max_new_tokens=512)
        generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        
        # Extract and parse the JSON from the model's response
        raw_output = generated_texts[0]
        try:
            # The model's output might include the prompt. We find the first '{' to start parsing.
            json_str = raw_output[raw_output.find('{'):raw_output.rfind('}')+1]
            return json.loads(json_str)
        except json.JSONDecodeError:
            logger.error(f"Failed to decode JSON from model output for frame {frame_path}")
            logger.error(f"Raw Output: {raw_output}")
            return None

def find_video_shots(video_path: str) -> list:
    """Uses PySceneDetect to find all shots in a video."""
    video = open_video(video_path)
    scene_manager = SceneManager()
    # Using a threshold of 27 is a good default for general content
    scene_manager.add_detector(ContentDetector(threshold=27))
    scene_manager.detect_scenes(video, show_progress=True)
    shot_list = scene_manager.get_scene_list()
    
    return [{
        "shot_number": i + 1,
        "start_seconds": shot[0].get_seconds(),
        "end_seconds": shot[1].get_seconds()
    } for i, shot in enumerate(shot_list)]

def extract_keyframe(video_capture, shot_info: dict, output_path: str) -> bool:
    """Extracts the middle frame of a shot and saves it."""
    middle_timestamp_msec = (shot_info["start_seconds"] + (shot_info["end_seconds"] - shot_info["start_seconds"]) / 2) * 1000
    video_capture.set(cv2.CAP_PROP_POS_MSEC, middle_timestamp_msec)
    success, frame = video_capture.read()
    if success:
        cv2.imwrite(output_path, frame)
    return success

if __name__ == "__main__":
    # --- Configuration ---
    video_path = "sample_videos/Hair Love.mp4"
    temp_frame_dir = "shot_keyframes"
    output_json_file = "hair_love_metadata.json"

    # --- Initialization ---
    os.makedirs(temp_frame_dir, exist_ok=True)
    
    annotator = Gemma3nShotAnnotator()
    shots = find_video_shots(video_path)
    video_capture = cv2.VideoCapture(video_path)
    
    full_metadata = []

    # --- Main Processing Loop ---
    for shot in shots:
        logger.info(f"--- Processing Shot {shot['shot_number']} ({shot['start_seconds']:.2f}s - {shot['end_seconds']:.2f}s) ---")
        frame_path = os.path.join(temp_frame_dir, f"shot_{shot['shot_number']}.jpg")
        
        if extract_keyframe(video_capture, shot, frame_path):
            json_metadata = annotator.generate_json_for_frame(frame_path)
            
            if json_metadata:
                # Add the shot timing information to the generated JSON
                json_metadata['shot_id'] = shot['shot_number']
                json_metadata['start_time_seconds'] = shot['start_seconds']
                json_metadata['end_time_seconds'] = shot['end_seconds']
                full_metadata.append(json_metadata)
                logger.info(f"Successfully generated metadata for shot {shot['shot_number']}")
        else:
            logger.warning(f"Failed to extract keyframe for shot {shot['shot_number']}.")

    # --- Cleanup and Save ---
    video_capture.release()
    with open(output_json_file, "w") as f:
        json.dump(full_metadata, f, indent=4)

    logger.info(f"\n✅ Processing complete. Full metadata saved to {output_json_file}")