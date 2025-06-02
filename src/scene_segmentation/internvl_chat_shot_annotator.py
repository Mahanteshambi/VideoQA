# video_analysis_project/src/scene_segmentation/internvl_chat_shot_annotator.py (Revised Snippets)

import torch
from PIL import Image
import numpy as np
from pathlib import Path
import logging
import moviepy.editor as mp
# Ensure you have the right AutoModel class, AutoTokenizer, and potentially a specific Image Processor if needed
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoImageProcessor # Or specific InternVL processor

logger = logging.getLogger(__name__)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class InternVLChatShotAnnotator:
    def __init__(self, model_checkpoint: str = "OpenGVLab/InternVL_2_5_HiCo_R16",
                 device: str = DEVICE):
        self.device = device
        self.model_checkpoint = model_checkpoint
        
        try:
            logger.info(f"Loading InternVL tokenizer from: {self.model_checkpoint}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint, trust_remote_code=True)

            logger.info(f"Loading InternVL model from: {self.model_checkpoint}")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_checkpoint,
                torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                low_cpu_mem_usage=True,
                trust_remote_code=True # Often required for models with custom code
            ).to(self.device)
            self.model.eval()

            # InternVL models often have an associated image processor for their ViT
            # This might be loaded separately or be part of the model/tokenizer object.
            # Let's assume we can load it or the model handles PIL images directly in .chat()
            # If a separate image_processor is needed:
            try:
                self.image_processor = AutoImageProcessor.from_pretrained(self.model_checkpoint, trust_remote_code=True)
                logger.info(f"InternVL Image Processor loaded successfully for {self.model_checkpoint}.")
            except Exception:
                logger.warning(f"Could not load a separate AutoImageProcessor for {self.model_checkpoint}. "
                               f"Assuming model.chat() handles PIL images or pixel_values are prepared differently.")
                self.image_processor = None


            logger.info(f"InternVL model '{self.model_checkpoint}' loaded successfully on {self.device}.")

        except Exception as e:
            logger.error(f"Failed to load InternVL model, tokenizer, or processor: {e}", exc_info=True)
            raise

    def _prepare_shot_frames_and_pixel_values(
        self, 
        original_video_path: str, 
        shot_info: dict, 
        num_keyframes: int = 3 # Adjust based on model's typical input
    ) -> tuple[list[Image.Image] | None, torch.Tensor | None]:
        """
        Extracts PIL Images for keyframes and prepares 'pixel_values' tensor if image_processor is available.
        The user's snippet shows 'pixel_values' being passed to model.chat().
        """
        frames_pil_list = [] # Will store PIL images
        video_clip = None
        start_sec, end_sec = shot_info["start_time_seconds"], shot_info["end_time_seconds"]
        duration_sec = end_sec - start_sec

        if duration_sec <= 0.01: timepoints = [start_sec] if num_keyframes > 0 else []
        elif num_keyframes == 1: timepoints = [start_sec + duration_sec / 2.0]
        else: timepoints = np.linspace(start_sec, end_sec, num_keyframes, endpoint=True)
        
        if not timepoints: return None, None

        try:
            video_clip = mp.VideoFileClip(original_video_path)
            for t in timepoints:
                frame_np = video_clip.get_frame(t)
                frames_pil_list.append(Image.fromarray(frame_np).convert("RGB"))
            
            if not frames_pil_list:
                return None, None

            # Prepare pixel_values using the image_processor if available
            # InternVL's model.chat() might take PIL images directly OR precomputed pixel_values.
            # The snippet `model.chat(tokenizer, pixel_values, question, ...)` implies pixel_values are precomputed.
            pixel_values_tensor = None
            if self.image_processor:
                # InternVL's image processor might expect a single image or a list for batching.
                # For a shot, we might represent it with one "best" keyframe or a sequence.
                # If the model's .chat takes pixel_values for a single "visual context moment":
                image_to_process = frames_pil_list[len(frames_pil_list) // 2] # Middle frame as representative
                processed_inputs = self.image_processor(images=image_to_process, return_tensors="pt").to(self.device)
                pixel_values_tensor = processed_inputs.pixel_values
            else:
                # If no separate image_processor, model.chat() might take PIL images directly.
                # This part needs to align with how your specific model.chat() expects visual input.
                logger.warning("No separate image_processor loaded for InternVL. "
                               "Relying on model.chat() to handle PIL images if supported, "
                               "or pixel_values might be missing if required by the snippet's format.")


            return frames_pil_list, pixel_values_tensor

        except Exception as e:
            logger.error(f"Error preparing frames/pixel_values for shot {shot_info.get('shot_number', 'N/A')} (InternVL): {e}", exc_info=True)
            return None, None
        finally:
            if video_clip: video_clip.close()

    def _generate_text_internvl(
        self, 
        pixel_values: torch.Tensor | None, # Expected by user's snippet
        pil_images_for_model_chat: list[Image.Image] | None, # Alternative if model.chat takes PIL images
        instruction: str,
        generation_config: dict = None, # From user's snippet
        num_patches_list: list = None,  # From user's snippet, may be specific to InternVL
        max_new_tokens: int = 150
    ) -> str | None:
        if not self.model or not self.tokenizer:
            logger.error("InternVL model or tokenizer not loaded.")
            return None

        # The user's snippet: model.chat(tokenizer, pixel_values, question, generation_config, num_patches_list=num_patches_list, ...)
        # This implies `pixel_values` is a primary input.
        # Some InternVL `chat` methods take `image` (single PIL) or `images` (list of PILs).
        # We need to match the exact signature.

        # Let's assume `generation_config` is a dict of parameters for `.generate()`
        # (e.g., num_beams, do_sample, temperature, etc.)
        # If not provided, use some defaults.
        gen_config_params = generation_config if generation_config else {}
        if 'max_new_tokens' not in gen_config_params: # Ensure max_new_tokens is included
             gen_config_params['max_new_tokens'] = max_new_tokens
        if 'do_sample' not in gen_config_params: # For more deterministic output initially
            gen_config_params['do_sample'] = False


        try:
            # Based on your snippet: model.chat(tokenizer, pixel_values, question, generation_config, num_patches_list=num_patches_list, ...)
            # The `question` is our `instruction`.
            # `pixel_values` should come from `_prepare_shot_frames_and_pixel_values`.
            # `num_patches_list` seems specific; if not generated by image_processor, its origin needs clarification.
            # For now, let's pass it as None if not available.
            
            # If model.chat() directly takes PIL images (common for some InternVL versions):
            # image_to_use = pil_images_for_model_chat[len(pil_images_for_model_chat)//2] if pil_images_for_model_chat else None
            # if image_to_use:
            #     response, _ = self.model.chat(
            #         image=image_to_use,
            #         question=instruction,
            #         tokenizer=self.tokenizer,
            #         **gen_config_params 
            #     )
            # else: 
            #     logger.warning("No PIL image provided for InternVL chat method that expects one.")
            #     return None

            # Aligning with your snippet which uses `pixel_values`:
            if pixel_values is not None:
                # The `video_prefix` you mentioned ("question = video_prefix + question1")
                # implies the textual prompt itself might need a special prefix if the model
                # doesn't use something like <image> or <video> within the prompt string directly.
                # The `tokenizer` is passed to `model.chat` - this suggests it might handle tokenizing the question.
                # Let's assume the `question` argument in `model.chat` takes the raw instruction string.
                
                # The `num_patches_list` argument is specific. If the image_processor doesn't return it,
                # we might omit it or need to figure out how it's derived.
                # Let's try calling without it first if it's optional.
                
                # Check the exact signature of `model.chat` from the specific InternVL class.
                # For now, constructing a call based on your snippet:
                output_text, _ = self.model.chat(
                    tokenizer=self.tokenizer,
                    pixel_values=pixel_values.to(self.device), # Ensure it's on the correct device
                    question=instruction,
                    generation_config=generation_config, # User provided this
                    # num_patches_list=num_patches_list, # Pass if available and required
                    history=None,
                    return_history=False # We only need the current response
                )
            else:
                logger.warning("Pixel values not available for InternVL generation.")
                return None
                
            return output_text.strip()

        except Exception as e:
            logger.error(f"Error during InternVL text generation: {e}", exc_info=True)
            return None

    # ... (get_rich_description, get_actions, get_genre_mood_cues methods would call _generate_text_internvl)
    # They remain conceptually the same, just passing different `instruction` strings.

    def get_rich_description(self, pixel_values, pil_images, shot_number: int) -> str | None:
        instruction = "Describe this video segment in detail. Focus on main subjects, objects, actions, and overall setting."
        logger.info(f"Generating rich description for shot {shot_number} using InternVL...")
        return self._generate_text_internvl(pixel_values, pil_images, instruction, max_new_tokens=150)

    def get_actions(self, pixel_values, pil_images, shot_number: int) -> list[str] | None:
        instruction = "List the primary actions or events occurring in this video segment. If multiple, separate them with commas."
        logger.info(f"Identifying actions for shot {shot_number} using InternVL...")
        generated_text = self._generate_text_internvl(pixel_values, pil_images, instruction, max_new_tokens=75)
        if generated_text:
            actions = [action.strip() for action in generated_text.split(',') if action.strip() and action.strip().lower() not in ["none", "n/a"]]
            return actions if actions else None
        return None

    def get_genre_mood_cues(self, pixel_values, pil_images, shot_number: int) -> list[str] | None:
        instruction = ("Analyze this video segment. Describe its overall mood. "
                       "List visual elements (colors, lighting), character expressions, or implied events contributing to this mood. "
                       "Also, list keywords suggesting a potential genre. Separate all items with commas.")
        logger.info(f"Extracting genre/mood cues for shot {shot_number} using InternVL...")
        generated_text = self._generate_text_internvl(pixel_values, pil_images, instruction, max_new_tokens=100)
        if generated_text:
            cues = [cue.strip() for cue in generated_text.split(',') if cue.strip() and cue.strip().lower() not in ["none", "n/a"]]
            return cues if cues else None
        return None


    def extract_metadata_for_shot(
        self, 
        original_video_path: str, 
        shot_info: dict, 
        num_keyframes_for_vllm: int = 3 # Number of frames to sample from the shot
    ) -> dict | None:
        if not self.model or not self.tokenizer:
            return {"error": "InternVL not initialized."}

        logger.info(f"Preparing InternVL input for shot {shot_info['shot_number']}...")
        # _prepare_shot_frames_and_pixel_values now returns a tuple
        frames_pil_list, pixel_values_tensor = self._prepare_shot_frames_and_pixel_values(
            original_video_path, 
            shot_info, 
            num_keyframes=num_keyframes_for_vllm
        )

        if pixel_values_tensor is None and not frames_pil_list: # Check if frame prep failed
            return {"error": f"Frame preparation failed for shot {shot_info['shot_number']}"}
        
        # Pass both to _generate_text_internvl; it will use what it needs based on model.chat signature
        description = self.get_rich_description(pixel_values_tensor, frames_pil_list, shot_info['shot_number'])
        actions = self.get_actions(pixel_values_tensor, frames_pil_list, shot_info['shot_number'])
        genre_mood_cues = self.get_genre_mood_cues(pixel_values_tensor, frames_pil_list, shot_info['shot_number'])
        
        return {
            "vllm_description": description,
            "vllm_actions": actions if actions else [],
            "vllm_genre_mood_keywords": genre_mood_cues if genre_mood_cues else [],
        }