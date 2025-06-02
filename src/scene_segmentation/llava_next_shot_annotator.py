# video_analysis_project/src/scene_segmentation/llava_next_shot_annotator.py (or your chosen annotator)

import torch
from PIL import Image
import numpy as np
from pathlib import Path
import logging
import moviepy.editor as mp
from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration
import json # For parsing the output

logger = logging.getLogger(__name__)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# The UNIFIED_JSON_EXTRACTION_PROMPT_INSTRUCTION defined above would go here
UNIFIED_JSON_EXTRACTION_PROMPT_INSTRUCTION = """Analyze the provided video shot carefully. Your task is to extract comprehensive metadata based SOLELY on the visual and implied content of THIS SHOT.
Respond ONLY with a single, valid JSON object. Do not add any explanatory text before or after the JSON.
Adhere strictly to the following JSON schema and instructions for each field:

{
  "GenreCues": [{ "genre_hint": "string (e.g., Action, Horror, Comedy, Romance, Sci-Fi)", "prominence_in_shot": "integer (0-100, how strongly this shot suggests the genre)" }, ...],
  "SubgenreCues": ["string (e.g., Sci-fi thriller, Romantic comedy, Dark fantasy)", ...],
  "AdjectiveTheme": ["string (e.g., Betrayal, Survival, Coming-of-age, Mysterious, Uplifting)", ...],
  "Mood": ["string (e.g., Suspenseful, Joyful, Tense, Melancholic, Energetic, Calm)", ...],
  "SettingContext": ["string (e.g., Urban street - night, Dense forest - daytime, Office interior, School classroom, Futuristic city)", ...],
  "ContentDescriptors": ["string (e.g., Dialogue-heavy, Fast-paced editing, Slow motion, Archival footage, Character close-up, Wide establishing shot, CGI effects)", ...],
  "LocationHints_Regional": ["string (e.g., Specific non-major regions like 'Scottish Highlands', 'Appalachian Trail', if clearly identifiable from this shot)", ...],
  "LocationHints_International": ["string (e.g., Recognizable international cities/landmarks like 'Paris Eiffel Tower', 'Tokyo Shibuya', 'Egyptian Pyramids', if clearly identifiable from this shot)", ...],
  "SearchKeywords": ["string (objects, specific actions, character types like 'elderly man' or 'person in red dress', distinct visual elements, concepts directly visible or strongly implied in this shot)", ...]
}

Detailed Instructions for Populating Fields (based ONLY on this shot):
- "GenreCues": Identify elements within *this shot* (visuals, implied actions, atmosphere) that *hint* at broader genres. Estimate prominence of these hints within this shot (0-100). If no strong cues, provide an empty list [].
- "SubgenreCues": List any specific subgenre types strongly suggested by *this shot's* content. Empty list if none.
- "AdjectiveTheme": List core themes, narrative concepts, or descriptive adjectives that best characterize *this shot*. Empty list if none.
- "Mood": Dominant emotional tone(s) clearly conveyed by *this shot*. Empty list if neutral or unclear.
- "SettingContext": Describe the primary location type and context (e.g., "Modern kitchen - bright morning", "Crowded marketplace - daytime", "Deserted alley - night"). Empty list if unclear.
- "ContentDescriptors": List specific markers describing the shot's nature, content type, or cinematic style (e.g., "POV shot", "Explosion", "Intense argument", "Nature documentary style"). Be comprehensive. Empty list if none.
- "LocationHints_Regional": List specific, identifiable local regions if evident. Empty list if generic or not applicable.
- "LocationHints_International": List specific, identifiable major international locations/landmarks if evident. Empty list if generic or not applicable.
- "SearchKeywords": Generate a rich list of diverse keywords for *this shot*. Include visible objects, specific actions, character types (e.g., "detective with magnifying glass", "children playing soccer"), setting details, and distinct visual elements. Aim for terms someone would use to find this specific shot. Empty list if nothing distinct.

Ensure all string values are properly escaped for JSON. If a category has no applicable information for this shot, use an empty list [] for array types.
"""


class LlavaNextShotAnnotator: # Or your chosen Annotator class
    def __init__(self, model_checkpoint: str = "llava-hf/llava-next-video-7b-hf", device: str = DEVICE):
        self.device = device
        self.model = None
        self.processor = None
        self.model_checkpoint = model_checkpoint
        try:
            logger.info(f"Loading LLaVA-NeXT-Video processor from: {self.model_checkpoint}")
            self.processor = LlavaNextVideoProcessor.from_pretrained(self.model_checkpoint)
            logger.info(f"Loading LLaVA-NeXT-Video model from: {self.model_checkpoint}")
            self.model = LlavaNextVideoForConditionalGeneration.from_pretrained(
                self.model_checkpoint, torch_dtype=torch.float16, low_cpu_mem_usage=True
            ).to(self.device)
            self.model.eval()
            logger.info(f"LLaVA-NeXT-Video model '{self.model_checkpoint}' loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load LLaVA-NeXT-Video model or processor: {e}", exc_info=True)
            raise

    def _prepare_shot_frames(
        self, original_video_path: str, shot_info: dict, num_frames_to_sample: int = 8
    ) -> list[Image.Image] | None:
        # This function remains the same as defined in the previous response
        # (extracts `num_frames_to_sample` PIL images from the shot)
        start_sec = shot_info["start_time_seconds"]
        end_sec = shot_info["end_time_seconds"]
        duration_sec = end_sec - start_sec

        if duration_sec <= 0.01:
            timepoints = [start_sec] if num_frames_to_sample > 0 else []
        elif num_frames_to_sample == 1:
            timepoints = [start_sec + duration_sec / 2.0]
        else:
            timepoints = np.linspace(start_sec, end_sec, num_frames_to_sample, endpoint=True)
        
        if not timepoints.size > 0 : return None

        frames_pil_list = []
        video_clip = None
        try:
            video_clip = mp.VideoFileClip(original_video_path)
            for t in timepoints:
                frame_np = video_clip.get_frame(t)
                frames_pil_list.append(Image.fromarray(frame_np).convert("RGB"))
            return frames_pil_list
        except Exception as e:
            logger.error(f"Error extracting frames for shot {shot_info.get('shot_number', 'N/A')}: {e}", exc_info=True)
            return None
        finally:
            if video_clip: video_clip.close()

    def _generate_json_metadata_from_frames(
        self, 
        frames: list[Image.Image], 
        max_new_tokens: int = 768 # Increased for potentially large JSON
    ) -> str | None:
        """Generates text using the LLaVA-NeXT-Video model with the unified JSON prompt."""
        if not self.model or not self.processor or not frames:
            logger.error("Model/processor not loaded or no frames provided for JSON generation.")
            return None

        try:
            # Use the UNIFIED_JSON_EXTRACTION_PROMPT_INSTRUCTION
            # LLaVA's prompt format typically is "USER: <video>\n[INSTRUCTION]\nASSISTANT:"
            # The instruction here is our detailed JSON request.
            prompt = f"USER: <video>\n{UNIFIED_JSON_EXTRACTION_PROMPT_INSTRUCTION}\nASSISTANT:"
            
            inputs = self.processor(text=prompt, videos=frames, return_tensors="pt").to(self.device)

            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            
            input_token_len = inputs.input_ids.shape[1]
            generated_text = self.processor.batch_decode(generated_ids[:, input_token_len:], skip_special_tokens=True)[0]
            
            return generated_text.strip()
        except Exception as e:
            logger.error(f"Error during LLaVA-NeXT-Video JSON generation: {e}", exc_info=True)
            return None

    def extract_metadata_for_shot(
        self, 
        original_video_path: str, 
        shot_info: dict, 
        num_keyframes_to_sample: int = 8 
    ) -> dict | None:
        """
        Orchestrates comprehensive JSON metadata extraction for a single shot using the VLLM.
        This replaces the separate get_rich_description, get_actions, etc. methods.
        """
        if not self.model or not self.processor:
            logger.error("LLaVA-NeXT-Video not initialized. Cannot extract metadata.")
            return {"error": "LLaVA-NeXT-Video not initialized.", "raw_vllm_output": None}

        logger.info(f"Preparing LLaVA-NeXT-Video input for shot {shot_info['shot_number']} for JSON metadata...")
        frames_pil_list = self._prepare_shot_frames(
            original_video_path, 
            shot_info, 
            num_frames_to_sample=num_keyframes_to_sample
        )

        if not frames_pil_list:
            logger.warning(f"Could not prepare frames for shot {shot_info['shot_number']}. Skipping VLLM JSON metadata.")
            return {"error": f"Frame preparation failed for shot {shot_info['shot_number']}", "raw_vllm_output": None}

        logger.info(f"Generating comprehensive JSON metadata for shot {shot_info['shot_number']}...")
        raw_generated_text = self._generate_json_metadata_from_frames(frames_pil_list, max_new_tokens=768) # Allow more tokens

        if raw_generated_text:
            try:
                # Basic cleanup to extract JSON block if VLLM wraps it
                json_text_cleaned = raw_generated_text
                if json_text_cleaned.strip().startswith("```json"):
                    json_text_cleaned = json_text_cleaned.split("```json")[1].split("```")[0].strip()
                elif json_text_cleaned.strip().startswith("```"):
                     json_text_cleaned = json_text_cleaned.split("```")[1].strip()
                
                # Find the first '{' and last '}' to isolate the JSON object robustly
                start_json = json_text_cleaned.find('{')
                end_json = json_text_cleaned.rfind('}')
                if start_json != -1 and end_json != -1 and end_json > start_json:
                    json_text_cleaned = json_text_cleaned[start_json : end_json+1]
                else:
                    logger.error(f"Could not find valid JSON delimiters in output for shot {shot_info['shot_number']}.")
                    raise json.JSONDecodeError("No valid JSON object found", json_text_cleaned, 0)

                metadata_json = json.loads(json_text_cleaned)
                logger.info(f"Successfully parsed JSON metadata for shot {shot_info['shot_number']}.")
                return {"vllm_generated_json_metadata": metadata_json, "raw_vllm_output": raw_generated_text}
            except json.JSONDecodeError as e:
                logger.error(f"Failed to decode JSON output for shot {shot_info['shot_number']}: {e}. Raw output: '{raw_generated_text}'")
                return {"error": "Failed to decode VLLM JSON output", "raw_vllm_output": raw_generated_text}
            except Exception as e_gen: # Catch other potential errors during parsing
                logger.error(f"Generic error parsing JSON for shot {shot_info['shot_number']}: {e_gen}. Raw output: '{raw_generated_text}'")
                return {"error": "Generic error parsing VLLM JSON output", "raw_vllm_output": raw_generated_text}
        else:
            logger.warning(f"VLLM returned no text for shot {shot_info['shot_number']}.")
            return {"error": "VLLM returned no text", "raw_vllm_output": None}