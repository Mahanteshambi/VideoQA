# video_analysis_project/src/scene_segmentation/internvl_3_1b_shot_annotator.py

import torch
from PIL import Image
import numpy as np
from pathlib import Path
import logging
import moviepy.editor as mp
from transformers import AutoModelForCausalLM, AutoTokenizer
from torchvision import transforms
import json

logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"InternVL3_1B_ShotAnnotator: Using device: {DEVICE}")

## CORRECTED ##: The full, unabridged prompt is included here.
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

# The rest of the file remains the same as the previously generated version...
def build_transform(input_size=448):
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform

def dynamic_preprocess(img: Image.Image, image_size: int, use_thumbnail: bool, max_num: int) -> list[Image.Image]:
    w, h = img.size
    if w == 0 or h == 0: return [img]
    scale = image_size / min(w, h)
    w_scaled, h_scaled = max(1, int(w * scale)), max(1, int(h * scale))
    img_resized = img.resize((w_scaled, h_scaled), Image.Resampling.LANCZOS)
    patches = []
    if use_thumbnail:
        num_patch_w = min(max_num, (w_scaled + image_size - 1) // image_size)
        num_patch_h = min(max_num, (h_scaled + image_size - 1) // image_size)
        for i in range(num_patch_h):
            for j in range(num_patch_w):
                x1, y1 = j * image_size, i * image_size
                x2, y2 = min(x1 + image_size, w_scaled), min(y1 + image_size, h_scaled)
                patch = img_resized.crop((x1, y1, x2, y2))
                if (x2 - x1) != image_size or (y2 - y1) != image_size:
                    patch = patch.resize((image_size, image_size), Image.Resampling.LANCZOS)
                patches.append(patch)
    else:
        if w_scaled < image_size or h_scaled < image_size:
            img_resized = img_resized.resize((image_size, image_size), Image.Resampling.LANCZOS)
        patches.append(img_resized)
    return patches if patches else [img_resized]

def get_frame_timepoints(start_time, end_time, num_segments=8):
    if num_segments <= 0: return np.array([])
    if num_segments == 1:
        return np.array([start_time + (end_time - start_time) / 2.0])
    return np.linspace(start_time, end_time, num_segments, endpoint=True)

class InternVL3_1B_ShotAnnotator:
    def __init__(self, model_checkpoint: str = "OpenGVLab/InternVL-C_1.3B-224px", device: str = DEVICE):
        self.device = device
        self.model = None
        self.tokenizer = None
        self.model_checkpoint = model_checkpoint
        self.image_input_size = 224

        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        try:
            logger.info(f"Loading InternVL tokenizer from: {self.model_checkpoint}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint, trust_remote_code=True)
            
            logger.info(f"Loading InternVL model from: {self.model_checkpoint} with dtype: {dtype}")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_checkpoint,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                attn_implementation="flash_attention_2"
            ).to(self.device)
            
            logger.info("Compiling the model with torch.compile()...")
            self.model = torch.compile(self.model, mode="reduce-overhead", fullgraph=True)

            self.model.eval()

            if hasattr(self.model, 'config') and hasattr(self.model.config, 'vision_config'):
                self.image_input_size = self.model.config.vision_config.image_size
                logger.info(f"Using image input size from model config: {self.image_input_size}")
            
            self.transform = build_transform(input_size=self.image_input_size)
            logger.info(f"InternVL model '{self.model_checkpoint}' loaded successfully on {self.device}.")

        except Exception as e:
            logger.error(f"Failed to load InternVL model or tokenizer from '{self.model_checkpoint}': {e}", exc_info=True)
            raise

    def _parse_and_validate_json(self, raw_text: str, shot_number: int) -> dict:
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
            
            return {"vllm_generated_json_metadata": metadata_json, "raw_vllm_output": raw_text}

        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON output for shot {shot_number} (InternVL): {e}. Raw output: '{raw_text}'")
            return {"error": "Failed to decode InternVL JSON output", "raw_vllm_output": raw_text}
        except Exception as e_gen:
            logger.error(f"Generic error parsing JSON for shot {shot_number} (InternVL): {e_gen}. Raw output: '{raw_text}'")
            return {"error": "Generic error parsing InternVL JSON output", "raw_vllm_output": raw_text}

    def _prepare_pixel_values_for_batch(
        self,
        original_video_path: str,
        shots_info: list[dict],
        num_segments: int,
        max_dynamic_patches: int
    ) -> tuple[list[torch.Tensor] | None, list[list[int]] | None, list[int] | None]:
        batch_pixel_values_list = []
        batch_num_patches_list = []
        valid_shot_indices = []
        video_clip = None

        try:
            video_clip = mp.VideoFileClip(original_video_path)
            for i, shot_info in enumerate(shots_info):
                timepoints = get_frame_timepoints(shot_info["start_time_seconds"], shot_info["end_time_seconds"], num_segments)
                if not timepoints.size > 0:
                    logger.warning(f"No timepoints to sample for shot {shot_info.get('shot_number', 'N/A')}")
                    continue

                shot_pixel_values_list = []
                shot_num_patches_list = []
                for t in timepoints:
                    frame_np = video_clip.get_frame(min(t, video_clip.duration - 0.001))
                    img_pil = Image.fromarray(frame_np).convert('RGB')
                    img_tiles_pil = dynamic_preprocess(img_pil, image_size=self.image_input_size, use_thumbnail=True, max_num=max_dynamic_patches)
                    
                    current_frame_pixel_values = torch.stack([self.transform(tile) for tile in img_tiles_pil])
                    shot_pixel_values_list.append(current_frame_pixel_values)
                    shot_num_patches_list.append(current_frame_pixel_values.shape[0])

                if shot_pixel_values_list:
                    batch_pixel_values_list.append(torch.cat(shot_pixel_values_list))
                    batch_num_patches_list.append(shot_num_patches_list)
                    valid_shot_indices.append(i)

            if not batch_pixel_values_list:
                return None, None, None

            return batch_pixel_values_list, batch_num_patches_list, valid_shot_indices

        except Exception as e:
            logger.error(f"Error processing video frames for batch: {e}", exc_info=True)
            return None, None, None
        finally:
            if video_clip: video_clip.close()

    @property
    def dtype(self):
        return next(self.model.parameters()).dtype

    def _generate_json_metadata_batch(
        self, 
        pixel_values_list: list[torch.Tensor],
        num_patches_list_batch: list[list[int]],
        instruction_prompt: str,
        max_new_tokens: int,
        generation_config_override: dict = None
    ) -> list[str]:
        if not self.model or not self.tokenizer:
            raise RuntimeError("InternVL model or tokenizer not loaded.")

        gen_config = {"max_new_tokens": max_new_tokens, "do_sample": False, "num_beams": 1}
        if generation_config_override:
            gen_config.update(generation_config_override)
        
        responses = []
        for pixel_values, num_patches_list in zip(pixel_values_list, num_patches_list_batch):
            try:
                image_placeholders = "".join([f"Frame {i+1} is <image>.\n" for i in range(len(num_patches_list))])
                full_question = image_placeholders + instruction_prompt

                response, _ = self.model.chat(
                    tokenizer=self.tokenizer,
                    pixel_values=pixel_values.to(self.device).to(self.dtype),
                    question=full_question,
                    generation_config=gen_config,
                    num_patches_list=num_patches_list,
                    history=None,
                    return_history=False
                )
                responses.append(response.strip())
            except Exception as e:
                logger.error(f"Error during one of the generations in a batch: {e}", exc_info=True)
                responses.append(json.dumps({"error": f"Generation failed: {e}"}))

        return responses

    def extract_metadata_for_batch(
        self, 
        original_video_path: str, 
        shots_info: list[dict], 
        num_keyframes_to_sample: int = 3,
        max_dynamic_patches_per_frame: int = 1
    ) -> list[dict]:
        if not self.model or not self.tokenizer:
            return [{"error": "InternVL model not initialized."}] * len(shots_info)

        pixel_values_list, num_patches_list_batch, valid_indices = self._prepare_pixel_values_for_batch(
            original_video_path, shots_info, num_keyframes_to_sample, max_dynamic_patches_per_frame
        )

        if pixel_values_list is None:
            return [{"error": "Pixel_values preparation failed for batch"}] * len(shots_info)

        raw_generated_texts = self._generate_json_metadata_batch(
            pixel_values_list=pixel_values_list,
            num_patches_list_batch=num_patches_list_batch,
            instruction_prompt=UNIFIED_JSON_EXTRACTION_PROMPT_INSTRUCTION,
            max_new_tokens=1024
        )

        final_results = []
        result_idx = 0
        for i in range(len(shots_info)):
            if i in valid_indices:
                shot_number = shots_info[i]['shot_number']
                parsed_json = self._parse_and_validate_json(raw_generated_texts[result_idx], shot_number)
                final_results.append(parsed_json)
                result_idx += 1
            else:
                final_results.append({"error": f"Shot was skipped during frame preparation", "raw_vllm_output": None})
        
        return final_results