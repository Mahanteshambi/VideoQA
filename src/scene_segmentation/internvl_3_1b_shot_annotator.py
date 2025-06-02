# video_analysis_project/src/scene_segmentation/internvl_3_1b_shot_annotator.py

import torch
from PIL import Image
import numpy as np
from pathlib import Path
import logging
import moviepy.editor as mp
from transformers import AutoModelForCausalLM, AutoTokenizer # For InternVL Chat
from torchvision import transforms # Used in your provided InternVL script
import json # For parsing the output

logger = logging.getLogger(__name__)

# --- Device Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"InternVL3_1B_ShotAnnotator: Using device: {DEVICE}")

# --- Unified Prompt (Copied from your LlavaNextShotAnnotator for consistency) ---
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
- "GenreCues": Identify elements within *this shot* (visuals, implied actions, atmosphere) that *hint* at broader genres. Estimate prominence of these hints within this shot (0-100). MUST be an array of objects, each with "genre_hint" and "prominence_in_shot". If none, use [].
- "SubgenreCues": List any specific subgenre types strongly suggested by *this shot's* content. MUST be an array of strings. If none, use [].
- "AdjectiveTheme": List core themes, narrative concepts, or descriptive adjectives that best characterize *this shot*. MUST be an array of strings. If none, use [].
- "Mood": Dominant emotional tone(s) clearly conveyed by *this shot*. MUST be an array of strings. If none, use [].
- "SettingContext": Describe the primary location type and context. MUST be an array of strings. If none, use [].
- "ContentDescriptors": List specific markers describing the shot's nature, content type, or cinematic style. MUST be an array of strings. If none, use [].
- "LocationHints_Regional": List specific, identifiable local regions if evident. MUST be an array of strings. If none, use [].
- "LocationHints_International": List specific, identifiable major international locations/landmarks if evident. MUST be an array of strings. If none, use [].
- "SearchKeywords": Generate a rich list of diverse keywords for *this shot*. MUST be an array of strings. If none, use [].

REMEMBER: 
- Every array field MUST be an array, even if it contains only one item or is empty. Empty arrays should be represented as [].
- ShotDescription MUST be a single string, not an array. If no description, use an empty string "".
"""

# --- Helper functions from your provided internvl_3_1b_shot_annotator.py ---
def build_transform(input_size=448): # Default to 448 if not specified by model config
    """Build transformation pipeline for images."""
    # This input_size should ideally come from the model's image processor config
    # For InternVL-C 1.3B, it's often 224 or 448.
    # The InternVL-Chat-V1.5 example uses dynamic sizing.
    # Let's assume a fixed size for simplicity here if dynamic_preprocess handles it.
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Standard ImageNet normalization
    ])
    return transform

def dynamic_preprocess(img: Image.Image, image_size: int, use_thumbnail: bool, max_num: int) -> list[Image.Image]:
    """Preprocess image with dynamic tiling. Returns a list of PIL Images (tiles)."""
    # (This function was provided in your internvl_3_1b_shot_annotator.py, keeping it)
    w, h = img.size
    if w == 0 or h == 0: return [img] # Avoid division by zero for empty images
    scale = image_size / min(w, h) if min(w,h) > 0 else 1.0
    
    # Ensure w_scaled and h_scaled are at least 1
    w_scaled, h_scaled = max(1, int(w * scale)), max(1, int(h * scale))
    img_resized = img.resize((w_scaled, h_scaled), Image.Resampling.LANCZOS)
    
    patches = []
    if use_thumbnail and (w_scaled > image_size or h_scaled > image_size):
        num_patch_w = min(max_num, (w_scaled + image_size - 1) // image_size)
        num_patch_h = min(max_num, (h_scaled + image_size - 1) // image_size)
        for i in range(num_patch_h):
            for j in range(num_patch_w):
                x1, y1 = j * image_size, i * image_size
                x2, y2 = min(x1 + image_size, w_scaled), min(y1 + image_size, h_scaled)
                patch = img_resized.crop((x1, y1, x2, y2))
                # If the patch is smaller than image_size, resize it
                if (x2 - x1) != image_size or (y2 - y1) != image_size:
                    patch = patch.resize((image_size, image_size), Image.Resampling.LANCZOS)
                patches.append(patch)
    else: # If not using thumbnailing or image is small enough
        # If image is smaller than target, resize up to target.
        if w_scaled < image_size or h_scaled < image_size:
            img_resized = img_resized.resize((image_size,image_size), Image.Resampling.LANCZOS)
        patches.append(img_resized)

    return patches if patches else [img_resized] # Ensure at least one image is returned


def get_frame_timepoints(start_time, end_time, num_segments=8): # Renamed from get_frame_indices
    """Get frame timepoints for video sampling."""
    # (This function was provided, fps is not used here, relies on time)
    if num_segments <= 0: return np.array([])
    if num_segments == 1:
        return np.array([start_time + (end_time - start_time) / 2.0])
    return np.linspace(start_time, end_time, num_segments, endpoint=True)

class InternVL3_1B_ShotAnnotator:
    def __init__(self, model_checkpoint: str = "OpenGVLab/InternVL-C_1.3B-224px", # Using a common 1.3B chat model
                 device: str = DEVICE):
        self.device = device
        self.model = None
        self.tokenizer = None
        self.model_checkpoint = model_checkpoint
        # Image size might be part of model's config, e.g., model.config.vision_config.image_size
        self.image_input_size = 224 # Default for many ViTs, InternVL-C_1.3B-224px implies this
        # Check model.config.image_size or model.config.vision_config.image_size after loading if available
        
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        try:
            logger.info(f"Loading InternVL tokenizer from: {self.model_checkpoint}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint, trust_remote_code=True)
            
            logger.info(f"Loading InternVL model from: {self.model_checkpoint} with dtype: {dtype}")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_checkpoint,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            ).to(self.device)
            self.model.eval()

            # Attempt to get image_size from model config if possible
            if hasattr(self.model, 'config') and hasattr(self.model.config, 'vision_config') and \
               hasattr(self.model.config.vision_config, 'image_size'):
                self.image_input_size = self.model.config.vision_config.image_size
                logger.info(f"Using image input size from model config: {self.image_input_size}")
            elif hasattr(self.model, 'config') and hasattr(self.model.config, 'image_size'): # Some models have it directly
                self.image_input_size = self.model.config.image_size
                logger.info(f"Using image input size from model config: {self.image_input_size}")
            else:
                logger.warning(f"Could not determine image_input_size from model config. Using default: {self.image_input_size}")

            self.transform = build_transform(input_size=self.image_input_size)
            
            logger.info(f"InternVL model '{self.model_checkpoint}' loaded successfully on {self.device}.")

        except Exception as e:
            logger.error(f"Failed to load InternVL model or tokenizer from '{self.model_checkpoint}': {e}", exc_info=True)
            raise

    def _prepare_pixel_values_for_shot(
        self,
        original_video_path: str,
        shot_info: dict,
        num_segments: int = 8, # Number of frames to sample for the shot representation
        max_dynamic_patches: int = 1 # Number of tiles per frame for dynamic_preprocess
    ) -> tuple[torch.Tensor | None, list[int] | None]:
        """Process video frames for a shot into pixel_values and num_patches_list."""
        video_clip = None
        try:
            video_clip = mp.VideoFileClip(original_video_path)
            timepoints = get_frame_timepoints(shot_info["start_time_seconds"], shot_info["end_time_seconds"], num_segments)
            
            if not timepoints.size > 0:
                logger.warning(f"No timepoints to sample for shot {shot_info.get('shot_number', 'N/A')}")
                return None, None

            all_pixel_values_list = []
            num_patches_list = []
            
            for t in timepoints:
                frame_np = video_clip.get_frame(min(t, video_clip.duration - 0.001 if video_clip.duration else t)) # Ensure t is within duration
                img_pil = Image.fromarray(frame_np).convert('RGB')
                
                # Dynamic preprocess returns a list of PIL image tiles
                img_tiles_pil = dynamic_preprocess(img_pil, image_size=self.image_input_size, use_thumbnail=True, max_num=max_dynamic_patches)
                
                # Transform each tile and stack them
                current_frame_pixel_values = torch.stack([self.transform(tile) for tile in img_tiles_pil])
                
                all_pixel_values_list.append(current_frame_pixel_values)
                num_patches_list.append(current_frame_pixel_values.shape[0]) # Number of patches for this frame (after tiling)
            
            if not all_pixel_values_list:
                logger.warning(f"No pixel values generated for shot {shot_info.get('shot_number', 'N/A')}")
                return None, None

            pixel_values = torch.cat(all_pixel_values_list).to(self.device)
            return pixel_values, num_patches_list
            
        except Exception as e:
            logger.error(f"Error processing video frames for shot {shot_info.get('shot_number', 'N/A')}: {e}", exc_info=True)
            return None, None # Return None for both on failure
        finally:
            if video_clip: video_clip.close()

    def _generate_json_metadata_with_internvl_chat(
        self, 
        pixel_values: torch.Tensor,
        num_patches_list: list[int],
        instruction_prompt: str, # This will be the UNIFIED_JSON_EXTRACTION_PROMPT_INSTRUCTION
        max_new_tokens: int = 1024, # Increased for large JSON
        generation_config_override: dict = None
    ) -> str | None:
        if not self.model or not self.tokenizer:
            logger.error("InternVL model or tokenizer not loaded.")
            return None
        
        # InternVL's model.chat expects a 'question' which is the text part of the prompt.
        # The visual context (pixel_values) is passed separately.
        # The `UNIFIED_JSON_EXTRACTION_PROMPT_INSTRUCTION` forms the `question`.
        # The model's `chat` method needs to be compatible with `pixel_values` and `num_patches_list`.

        # The original InternVL example had: `video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])`
        # `question = video_prefix + instruction`
        # This implies the instruction needs to be prefixed if multiple "frames" (sets of patches) are processed.
        # Here, `pixel_values` is a concatenation of all patches from all sampled keyframes.
        # `num_patches_list` indicates how many patches belong to each original keyframe.
        # The `model.chat` of InternVL is designed to handle this.

        # Construct the question for the chat model
        # If num_patches_list indicates multiple "image contexts" (from multiple keyframes)
        # we might need the <image> placeholders.
        # Assuming num_patches_list corresponds to groups of patches per "image" (keyframe) passed.
        image_placeholders = ""
        if len(num_patches_list) > 1 : # If we sampled multiple keyframes, and each generated patches
             image_placeholders = "".join([f"Frame {i+1} is <image>.\n" for i in range(len(num_patches_list))])
        elif len(num_patches_list) == 1 and num_patches_list[0] > 1: # Single keyframe, multiple tiles
            image_placeholders = "The following image is composed of multiple tiles <image>.\n"
        elif len(num_patches_list) == 1 and num_patches_list[0] == 1: # Single keyframe, single tile
            image_placeholders = "Consider the following image <image>.\n"


        full_question = image_placeholders + instruction_prompt

        gen_config = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False, # For more deterministic JSON
            "num_beams": 1,     # Beam search can sometimes mess up strict JSON
            # "temperature": 0.7, # Example if do_sample=True
            # "top_p": 0.9,       # Example if do_sample=True
        }
        if generation_config_override:
            gen_config.update(generation_config_override)

        try:
            # The model.chat() method might return just the response without history
            response = self.model.chat(
                tokenizer=self.tokenizer,
                pixel_values=pixel_values.to(self.dtype), # Ensure dtype matches model
                question=full_question,
                generation_config=gen_config, # Pass generation config
                num_patches_list=num_patches_list,
                history=None,
                return_history=False
            )
            
            # Handle different possible return types from model.chat()
            if isinstance(response, tuple):
                # If it returns (response, history)
                return response[0].strip()
            elif isinstance(response, dict) and 'response' in response:
                # If it returns a dict with 'response' key
                return response['response'].strip()
            elif isinstance(response, str):
                # If it returns just the string response
                return response.strip()
            else:
                logger.error(f"Unexpected response type from model.chat(): {type(response)}")
                return None
        except Exception as e:
            logger.error(f"Error during InternVL text generation with chat method: {e}", exc_info=True)
            return None
    
    @property
    def dtype(self): # Helper to get model's dtype
        return next(self.model.parameters()).dtype


    def extract_metadata_for_shot(
        self, 
        original_video_path: str, 
        shot_info: dict, 
        num_keyframes_to_sample: int = 3, # Sample a few frames, e.g., start, middle, end
        max_dynamic_patches_per_frame: int = 1 # How many tiles per frame (1 means no extra tiling if frame fits)
    ) -> dict: # Changed return type to always be dict for consistency
        """
        Orchestrates comprehensive JSON metadata extraction for a single shot using InternVL.
        """
        if not self.model or not self.tokenizer:
            logger.error("InternVL model not initialized. Cannot extract metadata.")
            return {"error": "InternVL model not initialized.", "raw_vllm_output": None}

        logger.info(f"Preparing InternVL input for shot {shot_info['shot_number']}...")
        pixel_values, num_patches_list = self._prepare_pixel_values_for_shot(
            original_video_path, 
            shot_info, 
            num_segments=num_keyframes_to_sample,
            max_dynamic_patches=max_dynamic_patches_per_frame
        )

        if pixel_values is None or num_patches_list is None:
            logger.warning(f"Could not prepare pixel_values for shot {shot_info['shot_number']}. Skipping VLLM JSON metadata.")
            return {"error": f"Pixel_values preparation failed for shot {shot_info['shot_number']}", "raw_vllm_output": None}

        logger.info(f"Generating comprehensive JSON metadata for shot {shot_info['shot_number']} using InternVL...")
        # The UNIFIED_JSON_EXTRACTION_PROMPT_INSTRUCTION will be passed as the 'instruction' to the generation method
        raw_generated_text = self._generate_json_metadata_with_internvl_chat(
            pixel_values=pixel_values,
            num_patches_list=num_patches_list,
            instruction_prompt=UNIFIED_JSON_EXTRACTION_PROMPT_INSTRUCTION,
            max_new_tokens=1024 # Allow ample tokens for JSON
        )

        if raw_generated_text:
            try:
                json_text_cleaned = raw_generated_text
                if json_text_cleaned.strip().startswith("```json"):
                    json_text_cleaned = json_text_cleaned.split("```json")[1].split("```")[0].strip()
                elif json_text_cleaned.strip().startswith("```"):
                     json_text_cleaned = json_text_cleaned.split("```")[1].strip()
                
                start_json = json_text_cleaned.find('{')
                end_json = json_text_cleaned.rfind('}')
                if start_json != -1 and end_json != -1 and end_json > start_json:
                    json_text_cleaned = json_text_cleaned[start_json : end_json+1]
                else:
                    raise json.JSONDecodeError("No valid JSON object found", json_text_cleaned, 0)

                metadata_json = json.loads(json_text_cleaned)
                
                # --- Start: JSON Validation and Defaulting Logic (copied from LLaVA annotator) ---
                required_fields = {
                    "ShotDescription": str, "GenreCues": list, "SubgenreCues": list,
                    "AdjectiveTheme": list, "Mood": list, "SettingContext": list,
                    "ContentDescriptors": list, "LocationHints_Regional": list,
                    "LocationHints_International": list, "SearchKeywords": list
                }
                validated_metadata = {}
                for field, expected_type in required_fields.items():
                    value = metadata_json.get(field)
                    if value is None: # Field missing
                        validated_metadata[field] = "" if expected_type == str else []
                    elif isinstance(value, expected_type):
                        validated_metadata[field] = value
                    else: # Type mismatch, try to coerce or default
                        logger.warning(f"Type mismatch for field '{field}' in shot {shot_info['shot_number']}. Expected {expected_type}, got {type(value)}. Value: {value}")
                        if expected_type == list and not isinstance(value, list):
                            validated_metadata[field] = [str(value)] if value else [] # Convert to list of strings if not None
                        elif expected_type == str and isinstance(value, list):
                            validated_metadata[field] = str(value[0]) if value else ""
                        else: # Default if coercion is tricky
                            validated_metadata[field] = "" if expected_type == str else []
                
                # Specific validation for GenreCues items
                if "GenreCues" in validated_metadata and isinstance(validated_metadata["GenreCues"], list):
                    processed_genre_cues = []
                    for cue in validated_metadata["GenreCues"]:
                        if isinstance(cue, dict) and "genre_hint" in cue and "prominence_in_shot" in cue:
                            try: # Ensure prominence is int
                                cue["prominence_in_shot"] = int(cue["prominence_in_shot"])
                                processed_genre_cues.append(cue)
                            except ValueError:
                                logger.warning(f"Invalid prominence value in GenreCue for shot {shot_info['shot_number']}: {cue}. Using default.")
                                processed_genre_cues.append({"genre_hint": str(cue.get("genre_hint","Unknown")), "prominence_in_shot": 50})
                        elif isinstance(cue, str): # Handle if model just gives a string
                             processed_genre_cues.append({"genre_hint": cue, "prominence_in_shot": 50})
                        # else: skip malformed dict
                    validated_metadata["GenreCues"] = processed_genre_cues
                # --- End: JSON Validation and Defaulting Logic ---

                logger.info(f"Successfully parsed and validated JSON metadata for shot {shot_info['shot_number']} using InternVL.")
                return {"vllm_generated_json_metadata": validated_metadata, "raw_vllm_output": raw_generated_text}
            except json.JSONDecodeError as e:
                logger.error(f"Failed to decode JSON output for shot {shot_info['shot_number']} (InternVL): {e}. Raw output: '{raw_generated_text}'")
                return {"error": "Failed to decode InternVL JSON output", "raw_vllm_output": raw_generated_text}
            except Exception as e_gen:
                logger.error(f"Generic error parsing JSON for shot {shot_info['shot_number']} (InternVL): {e_gen}. Raw output: '{raw_generated_text}'")
                return {"error": "Generic error parsing InternVL JSON output", "raw_vllm_output": raw_generated_text}
        else:
            logger.warning(f"InternVL returned no text for shot {shot_info['shot_number']}.")
            return {"error": "InternVL returned no text", "raw_vllm_output": None}