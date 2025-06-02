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

Detailed Instructions for Populating Fields (based ONLY on this shot):
- "ShotDescription": Provide a clear, natural language description of what is happening in this shot. Focus on key visual elements, actions, and atmosphere. Be specific but concise.
- "GenreCues": Identify elements within *this shot* (visuals, implied actions, atmosphere) that *hint* at broader genres. Estimate prominence of these hints within this shot (0-100). MUST be an array of objects, each with "genre_hint" and "prominence_in_shot".
- "SubgenreCues": List any specific subgenre types strongly suggested by *this shot's* content. MUST be an array of strings.
- "AdjectiveTheme": List core themes, narrative concepts, or descriptive adjectives that best characterize *this shot*. MUST be an array of strings.
- "Mood": Dominant emotional tone(s) clearly conveyed by *this shot*. MUST be an array of strings.
- "SettingContext": Describe the primary location type and context. MUST be an array of strings.
- "ContentDescriptors": List specific markers describing the shot's nature, content type, or cinematic style. MUST be an array of strings.
- "LocationHints_Regional": List specific, identifiable local regions if evident. MUST be an array of strings.
- "LocationHints_International": List specific, identifiable major international locations/landmarks if evident. MUST be an array of strings.
- "SearchKeywords": Generate a rich list of diverse keywords for *this shot*. MUST be an array of strings.

REMEMBER: 
- Every array field MUST be an array, even if it contains only one item or is empty. Empty arrays should be represented as [].
- ShotDescription MUST be a single string, not an array.

Example of proper formatting:
- String field: "ShotDescription": "A lone figure walks through a dimly lit urban alley at night, casting long shadows on wet pavement"
- Single item array: ["Urban street - night"]
- Multiple items: ["Fast-paced editing", "Wide establishing shot"]
- Empty array: []
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
                # Find the first '{' and last '}' to isolate the JSON object robustly
                start_json = raw_generated_text.find('{')
                end_json = raw_generated_text.rfind('}')
                if start_json != -1 and end_json != -1 and end_json > start_json:
                    json_text_cleaned = raw_generated_text[start_json : end_json+1]
                else:
                    logger.error(f"Could not find valid JSON delimiters in output for shot {shot_info['shot_number']}.")
                    raise json.JSONDecodeError("No valid JSON object found", raw_generated_text, 0)

                try:
                    metadata_json = json.loads(json_text_cleaned)
                    
                    # Validate required fields and types
                    required_fields = {
                        "ShotDescription": str,
                        "GenreCues": list,
                        "SubgenreCues": list,
                        "AdjectiveTheme": list,
                        "Mood": list,
                        "SettingContext": list,
                        "ContentDescriptors": list,
                        "LocationHints_Regional": list,
                        "LocationHints_International": list,
                        "SearchKeywords": list
                    }

                    # Validate all required fields exist and have correct types
                    for field, expected_type in required_fields.items():
                        if field not in metadata_json:
                            metadata_json[field] = "" if expected_type == str else []
                        elif not isinstance(metadata_json[field], expected_type):
                            if expected_type == list:
                                # Convert single items to list if needed
                                metadata_json[field] = [metadata_json[field]]
                            elif expected_type == str and isinstance(metadata_json[field], list):
                                # Convert single-item list to string if needed
                                metadata_json[field] = metadata_json[field][0] if metadata_json[field] else ""

                    # Validate GenreCues array items
                    if metadata_json["GenreCues"]:
                        for i, cue in enumerate(metadata_json["GenreCues"]):
                            if not isinstance(cue, dict) or "genre_hint" not in cue or "prominence_in_shot" not in cue:
                                # Fix malformed genre cues
                                if isinstance(cue, str):
                                    metadata_json["GenreCues"][i] = {
                                        "genre_hint": cue,
                                        "prominence_in_shot": 50  # Default to medium prominence
                                    }
                                else:
                                    # Remove invalid entries
                                    metadata_json["GenreCues"] = [gc for gc in metadata_json["GenreCues"] 
                                                                if isinstance(gc, dict) and "genre_hint" in gc 
                                                                and "prominence_in_shot" in gc]
                                    break

                    logger.info(f"Successfully parsed and validated JSON metadata for shot {shot_info['shot_number']}.")
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