# video_analysis_project/src/scene_segmentation/smolvlm_shot_annotator.py

import torch
from PIL import Image
import numpy as np
import logging
import moviepy.editor as mp
from transformers import AutoProcessor, AutoModel
import json
import traceback

logger = logging.getLogger(__name__)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# This is the same detailed prompt used by your other annotators.
# We will test SmolVLM's ability to adhere to this complex instruction.
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

class SmolVLMShotAnnotator:
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
            self.model = AutoModel.from_pretrained(
                self.model_checkpoint,
                torch_dtype=dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            ).to(self.device)
            
            logger.info("Compiling the SmolVLM model with torch.compile()...")
            self.model = torch.compile(self.model, mode="reduce-overhead", fullgraph=True)
            
            self.model.eval()
            logger.info(f"SmolVLM model '{self.model_checkpoint}' loaded successfully.")

        except Exception as e:
            logger.error(f"Failed to load SmolVLM model or processor: {e}", exc_info=True)
            raise

    def _prepare_shot_frames(self, original_video_path: str, shot_info: dict, num_frames_to_sample: int) -> list[Image.Image] | None:
        """Extracts PIL Images from a shot's duration."""
        start_sec, end_sec = shot_info["start_time_seconds"], shot_info["end_time_seconds"]
        timepoints = np.linspace(start_sec, end_sec, num_frames_to_sample) if end_sec > start_sec else [start_sec]
        
        frames_pil_list = []
        video_clip = None
        try:
            video_clip = mp.VideoFileClip(original_video_path)
            for t in timepoints:
                frame_np = video_clip.get_frame(t)
                frames_pil_list.append(Image.fromarray(frame_np).convert("RGB"))
            return frames_pil_list
        except Exception as e:
            logger.error(f"Error extracting frames for shot {shot_info.get('shot_number', 'N/A')}: {e}\n{traceback.format_exc()}")
            return None
        finally:
            if video_clip:
                video_clip.close()

    def _parse_and_validate_json(self, raw_text: str, shot_number: int) -> dict:
        """Cleans and validates the JSON output from the VLLM."""
        try:
            json_text_cleaned = raw_text
            if "```json" in json_text_cleaned:
                json_text_cleaned = json_text_cleaned.split("```json")[1].split("```")[0].strip()
            
            start_json = json_text_cleaned.find('{')
            end_json = json_text_cleaned.rfind('}')
            if start_json == -1 or end_json == -1:
                raise json.JSONDecodeError("No valid JSON object found", json_text_cleaned, 0)
            
            json_text_cleaned = json_text_cleaned[start_json : end_json + 1]
            metadata_json = json.loads(json_text_cleaned)
            
            # Simple validation can be added here if needed
            return {"vllm_generated_json_metadata": metadata_json, "raw_vllm_output": raw_text}
        except Exception as e:
            logger.error(f"Failed to decode JSON output for shot {shot_number}: {e}. Raw output: '{raw_text}'")
            return {"error": "Failed to decode VLLM JSON output", "raw_vllm_output": raw_text}

    def extract_metadata_for_batch(
        self, 
        original_video_path: str, 
        shots_info: list[dict], 
        num_keyframes_to_sample: int = 4 # SmolVLM is efficient, can use fewer frames
    ) -> list[dict]:
        """
        Orchestrates JSON metadata extraction for a BATCH of shots using SmolVLM.
        This is the primary method for high-performance processing.
        """
        if not self.model or not self.processor:
            return [{"error": "SmolVLM not initialized."}] * len(shots_info)

        # 1. Prepare inputs for the entire batch
        batch_messages = []
        video_clips_for_shots = []
        valid_indices = []

        for i, shot_info in enumerate(shots_info):
            frames = self._prepare_shot_frames(original_video_path, shot_info, num_keyframes_to_sample)
            if frames:
                # The processor expects a list of PIL images, which it treats as video frames
                message = [
                    {"role": "user", "content": [
                        {"type": "image"}, # Placeholder for video frames
                        {"type": "text", "text": UNIFIED_JSON_EXTRACTION_PROMPT_INSTRUCTION}
                    ]}
                ]
                batch_messages.append(message)
                video_clips_for_shots.append(frames)
                valid_indices.append(i)
            else:
                logger.warning(f"Skipping shot {shot_info['shot_number']} due to frame extraction failure.")

        if not batch_messages:
            return [{"error": "Frame extraction failed for all shots in batch"}] * len(shots_info)

        # 2. Process the entire batch with a single call
        try:
            # The processor handles the list of images as video frames
            inputs = self.processor.apply_chat_template(
                batch_messages,
                images=video_clips_for_shots, # Pass the list of frame lists
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            ).to(self.device)

            # 3. Generate responses for the entire batch
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=1024, do_sample=False)
            
            raw_generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        except Exception as e:
            logger.error(f"Critical error during SmolVLM batch generation: {e}\n{traceback.format_exc()}")
            return [{"error": f"Batch generation failed: {e}", "raw_vllm_output": None}] * len(shots_info)
        
        # 4. Parse and map results back to the original shots
        final_results = []
        result_idx = 0
        for i in range(len(shots_info)):
            if i in valid_indices:
                shot_number = shots_info[i]['shot_number']
                # The model's response often includes the prompt, so we clean it
                response_text = raw_generated_texts[result_idx].split("ASSISTANT:")[-1].strip()
                parsed_json = self._parse_and_validate_json(response_text, shot_number)
                final_results.append(parsed_json)
                result_idx += 1
            else:
                final_results.append({"error": "Shot was skipped during frame preparation", "raw_vllm_output": None})
        
        return final_results