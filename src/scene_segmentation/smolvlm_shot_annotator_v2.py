# scene_segmentation/smolvlm_shot_annotator_v2.py

import torch
from PIL import Image
import logging
import moviepy.editor as mp
# ## MODIFIED ##: Imported the specific model class instead of a generic AutoModel
from transformers import AutoProcessor, Idefics3ForConditionalGeneration
import json
import traceback
import os
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# This is the same detailed prompt. We are testing if SmolVLM can handle it.
UNIFIED_JSON_EXTRACTION_PROMPT_INSTRUCTION = """Analyze the provided video shot carefully. Your task is to extract comprehensive metadata based SOLELY on the visual and implied content of THIS SHOT.
Respond ONLY with a single, valid, properly formatted JSON object. Do not add any explanatory text before or after the JSON.
IMPORTANT JSON FORMATTING RULES:
1. Ensure all arrays are properly closed with square brackets []
2. All array items must be separated by commas
3. All JSON objects must be properly closed with curly braces {}
4. All field names must exactly match the schema - no variations allowed
5. All string values must be properly quoted
6. Use the exact field names as specified - do not modify or abbreviate them

Adhere strictly to the following JSON schema and instructions for each field:

{
  "ShotDescription": "string (A clear, concise description of what is happening in this shot, focusing on key visual elements, actions, and atmosphere)",
  "GenreCues": [{ "genre_hint": "string (e.g., Action, Horror, Comedy, Romance, Sci-Fi)", "prominence_in_shot": "integer (0-100, how strongly this shot suggests the genre)" }],
  "SubgenreCues": ["string (e.g., Sci-fi thriller, Romantic comedy, Dark fantasy)"],
  "AdjectiveTheme": ["string (e.g., Betrayal, Survival, Coming-of-age, Mysterious, Uplifting)"],
  "Mood": ["string (e.g., Suspenseful, Joyful, Tense, Melancholic, Energetic, Calm)"],
  "SettingContext": ["string (e.g., Urban street - night, Dense forest - daytime, Office interior, School classroom, Futuristic city)"],
  "ContentDescriptors": ["string (e.g., Dialogue-heavy, Fast-paced editing, Slow motion, Archival footage, Character close-up, Wide establishing shot, CGI effects)"],
  "LocationHints_Regional": ["string (e.g., Specific non-major regions like 'Scottish Highlands', 'Appalachian Trail', if clearly identifiable from this shot)"],
  "LocationHints_International": ["string (e.g., Recognizable international cities/landmarks like 'Paris Eiffel Tower', 'Tokyo Shibuya', 'Egyptian Pyramids', if clearly identifiable from this shot)"],
  "SearchKeywords": ["string (objects, specific actions, character types like 'elderly man' or 'person in red dress', distinct visual elements, concepts directly visible or strongly implied in this shot)"]
}
"""

class SmolVLMShotAnnotatorV2:
    def __init__(self, model_checkpoint: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct", device: str = DEVICE):
        self.device = device
        self.model = None
        self.processor = None
        self.model_checkpoint = model_checkpoint
        
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        try:
            logger.info(f"Loading SmolVLM processor from: {self.model_checkpoint}")
            self.processor = AutoProcessor.from_pretrained(self.model_checkpoint, trust_remote_code=True)
            
            logger.info(f"Loading SmolVLM model from: {self.model_checkpoint} with dtype: {dtype}")
            
            # ## MODIFIED ##: Replaced AutoModelForVision2Seq with the correct, specific class
            self.model = Idefics3ForConditionalGeneration.from_pretrained(
                self.model_checkpoint,
                torch_dtype=dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                # _attn_implementation="flash_attention_2"  # Use Flash Attention 2 if installed
            ).to(self.device)
            
            self.model.eval()
            logger.info(f"SmolVLM model '{self.model_checkpoint}' loaded successfully.")

        except Exception as e:
            logger.error(f"Failed to load SmolVLM model or processor: {e}", exc_info=True)
            raise

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
            
        temp_dir = Path("/tmp/smolvlm_shots")
        temp_dir.mkdir(exist_ok=True)
        temp_video_path = str(temp_dir / f"shot_{shot_info['shot_number']}_{uuid.uuid4()}.mp4")
        
        try:
            # Use moviepy to create a subclip of the shot
            with mp.VideoFileClip(original_video_path) as video:
                # ## MODIFIED ##: Removed the invalid 'temp_audiofile_path' argument
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

    def _parse_and_validate_json(self, raw_text: str, shot_number: int) -> dict:
        """Cleans and validates the JSON output from the VLLM."""
        try:
            start_json = raw_text.find('{')
            end_json = raw_text.rfind('}')
            if start_json == -1 or end_json == -1 or end_json < start_json:
                raise json.JSONDecodeError("No valid JSON object found in the model's output.", raw_text, 0)

            json_text_cleaned = raw_text[start_json : end_json + 1]
            metadata_json = json.loads(json_text_cleaned)
            
            return {"vllm_generated_json_metadata": metadata_json, "raw_vllm_output": raw_text}
        except Exception as e:
            logger.error(f"Failed to decode or parse JSON output for shot {shot_number}: {e}. Raw output: '{raw_text}'")
            return {"error": "Failed to parse VLLM JSON output", "raw_vllm_output": raw_text}

    def extract_metadata_for_shot(self, original_video_path: str, shot_info: dict, num_keyframes_to_sample: int = 4) -> dict:
        """
        Orchestrates metadata extraction for a SINGLE shot by passing its video segment directly to SmolVLM.
        """
        if not self.model or not self.processor:
            return {"error": "SmolVLM not initialized."}

        temp_video_path = None
        try:
            logger.info(f"Creating temporary video for shot {shot_info['shot_number']}...")
            temp_video_path = self._create_temp_shot_video(original_video_path, shot_info)
            if not temp_video_path:
                return {"error": "Failed to create temporary video for shot"}

            messages = [{
                "role": "user",
                "content": [
                    {"type": "video"},
                    {"type": "text", "text": UNIFIED_JSON_EXTRACTION_PROMPT_INSTRUCTION}
                ]
            }]
            
            inputs = self.processor.apply_chat_template(
                messages,
                video_paths=[temp_video_path],
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt"
            ).to(self.device)

            logger.info(f"Generating metadata for shot {shot_info['shot_number']}...")
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=1024, do_sample=False)
            
            response_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            cleaned_response = response_text.split("ASSISTANT:")[-1].strip()
            return self._parse_and_validate_json(cleaned_response, shot_info['shot_number'])

        except Exception as e:
            logger.error(f"A critical error occurred processing shot {shot_info['shot_number']} with SmolVLM: {e}", exc_info=True)
            return {"error": f"Critical failure in SmolVLM processing: {e}"}
        finally:
            if temp_video_path and os.path.exists(temp_video_path):
                try:
                    os.remove(temp_video_path)
                    logger.info(f"Cleaned up temporary file: {temp_video_path}")
                except OSError as e:
                    logger.error(f"Error removing temporary file {temp_video_path}: {e}")