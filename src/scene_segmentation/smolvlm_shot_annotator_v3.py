# scene_segmentation/smolvlm_shot_annotator_v2.py

import torch
from PIL import Image
import numpy as np
import logging
import moviepy.editor as mp
from transformers import AutoProcessor, Idefics3ForConditionalGeneration
import json
import traceback
import os
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
        
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

        if not torch.cuda.is_available():
            self.device = "mps" if torch.backends.mps.is_available() else "cpu"
            dtype = torch.float32 
            logger.info(f"CUDA not found. Setting device to '{self.device}' and dtype to '{dtype}' for Mac.")
        
        try:
            logger.info(f"Loading SmolVLM processor from: {self.model_checkpoint}")
            self.processor = AutoProcessor.from_pretrained(self.model_checkpoint, trust_remote_code=True)
            
            logger.info(f"Loading SmolVLM model from: {self.model_checkpoint} with dtype: {dtype}")
            self.model = Idefics3ForConditionalGeneration.from_pretrained(
                self.model_checkpoint,
                torch_dtype=dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            ).to(self.device)
            
            self.model.eval()
            logger.info(f"SmolVLM model '{self.model_checkpoint}' loaded successfully.")

        except Exception as e:
            logger.error(f"Failed to load SmolVLM model or processor: {e}", exc_info=True)
            raise

    # ## NEW ##: Re-introducing this helper method to extract frames manually.
    def _prepare_shot_frames(self, original_video_path: str, shot_info: dict, num_frames_to_sample: int) -> list[Image.Image] | None:
        """Extracts PIL Images from a shot's duration."""
        start_sec = shot_info["start_time_seconds"]
        end_sec = shot_info["end_time_seconds"]

        if end_sec <= start_sec:
            logger.warning(f"Shot {shot_info.get('shot_number')} has no duration. Skipping.")
            return None

        timepoints = np.linspace(start_sec, end_sec, num_frames_to_sample)
        
        frames_pil_list = []
        try:
            with mp.VideoFileClip(original_video_path) as video_clip:
                for t in timepoints:
                    frame_np = video_clip.get_frame(t)
                    frames_pil_list.append(Image.fromarray(frame_np).convert("RGB"))
            return frames_pil_list
        except Exception as e:
            logger.error(f"Error extracting frames for shot {shot_info.get('shot_number', 'N/A')}: {e}\n{traceback.format_exc()}")
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

    def extract_metadata_for_shot(self, original_video_path: str, shot_info: dict, num_keyframes_to_sample: int = 8) -> dict:
        """
        Orchestrates metadata extraction for a SINGLE shot by passing its frames directly to SmolVLM.
        """
        if not self.model or not self.processor:
            return {"error": "SmolVLM not initialized."}

        try:
            # 1. Manually extract frames for the shot
            logger.info(f"Preparing frames for shot {shot_info['shot_number']}...")
            frames = self._prepare_shot_frames(original_video_path, shot_info, num_keyframes_to_sample)
            if not frames:
                return {"error": "Failed to prepare frames for shot"}

            # 2. Prepare the prompt using the chat template and the extracted frames
            # Note: The placeholder "type" is now "image" since we are passing a list of images.
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image"}, 
                    {"type": "text", "text": UNIFIED_JSON_EXTRACTION_PROMPT_INSTRUCTION}
                ]
            }]
            
            # ## MODIFIED ##: Using the `images` argument instead of `video_paths`
            inputs = self.processor.apply_chat_template(
                messages,
                images=frames, # Pass the list of PIL images
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt"
            ).to(self.device)

            # 3. Generate the response
            logger.info(f"Generating metadata for shot {shot_info['shot_number']}...")
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=1024, do_sample=False)
            
            response_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # 4. Parse the output
            cleaned_response = response_text.split("ASSISTANT:")[-1].strip()
            return self._parse_and_validate_json(cleaned_response, shot_info['shot_number'])

        except Exception as e:
            logger.error(f"A critical error occurred processing shot {shot_info['shot_number']} with SmolVLM: {e}", exc_info=True)
            return {"error": f"Critical failure in SmolVLM processing: {e}"}