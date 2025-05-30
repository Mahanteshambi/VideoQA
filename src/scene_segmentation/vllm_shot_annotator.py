# video_analysis_project/src/scene_segmentation/vllm_shot_annotator.py

import torch
from PIL import Image
import numpy as np
from pathlib import Path
import logging
import moviepy.editor as mp
# Updated imports for LLaVA-NeXT-Video
from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration

logger = logging.getLogger(__name__)

# --- Device Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"VLLMShotAnnotator: Using device: {DEVICE}")

class VLLMShotAnnotator:
    def __init__(self, model_checkpoint: str = "llava-hf/llava-next-video-7b-hf", # Example checkpoint
                 device: str = DEVICE):
        """
        Initializes the LLaVA-NeXT-Video annotator.

        Args:
            model_checkpoint (str): Hugging Face model identifier for LLaVA-NeXT-Video.
            device (str): "cuda" or "cpu".
        """
        self.device = device
        self.model = None
        self.processor = None
        self.model_checkpoint = model_checkpoint

        try:
            logger.info(f"Loading LLaVA-NeXT-Video processor from: {self.model_checkpoint}")
            self.processor = LlavaNextVideoProcessor.from_pretrained(self.model_checkpoint)
            
            logger.info(f"Loading LLaVA-NeXT-Video model from: {self.model_checkpoint}")
            self.model = LlavaNextVideoForConditionalGeneration.from_pretrained(
                self.model_checkpoint, 
                torch_dtype=torch.float16, # Often recommended for these large models
                low_cpu_mem_usage=True,    # Helps with memory if loading large models
            ).to(self.device)
            
            self.model.eval()
            logger.info(f"LLaVA-NeXT-Video model '{self.model_checkpoint}' loaded successfully on {self.device}.")

        except Exception as e:
            logger.error(f"Failed to load LLaVA-NeXT-Video model or processor: {e}", exc_info=True)
            raise

    def _prepare_shot_frames(
        self, 
        original_video_path: str, 
        shot_info: dict, 
        num_frames_to_sample: int = 8 # LLaVA-NeXT-Video might have specific expectations for num_frames
    ) -> list[Image.Image] | None:
        """
        Extracts and returns a list of PIL Image objects for a given shot.
        """
        start_sec = shot_info["start_time_seconds"]
        end_sec = shot_info["end_time_seconds"]
        duration_sec = end_sec - start_sec

        # Ensure num_frames_to_sample is at least 1 if duration is very short but positive
        if duration_sec <= 0.01:
            logger.warning(f"Shot {shot_info.get('shot_number', 'N/A')} has negligible duration ({duration_sec:.3f}s). Attempting to sample 1 frame at start.")
            timepoints = [start_sec] if num_frames_to_sample > 0 else []
            actual_num_frames = 1 if timepoints else 0
        elif num_frames_to_sample == 1:
            timepoints = [start_sec + duration_sec / 2.0] # Middle frame
            actual_num_frames = 1
        else:
            # Sample frames, ensuring endpoint is included if duration allows
            timepoints = np.linspace(start_sec, end_sec, num_frames_to_sample, endpoint=True)
            actual_num_frames = len(timepoints)
        
        if actual_num_frames == 0:
            logger.warning(f"Shot {shot_info.get('shot_number', 'N/A')} resulted in 0 frames to sample.")
            return None

        frames_pil_list = []
        video_clip = None
        try:
            video_clip = mp.VideoFileClip(original_video_path)
            for t in timepoints:
                frame_np = video_clip.get_frame(t) # H, W, C (RGB)
                frames_pil_list.append(Image.fromarray(frame_np).convert("RGB"))
            
            return frames_pil_list
        except Exception as e:
            logger.error(f"Error extracting frames for shot {shot_info.get('shot_number', 'N/A')}: {e}", exc_info=True)
            return None
        finally:
            if video_clip:
                video_clip.close()


    def _generate_text_from_shot_frames(
        self, 
        frames: list[Image.Image], 
        prompt_instruction: str, # The actual question/instruction
        max_new_tokens: int = 100
    ) -> str | None:
        """
        Generates text from a list of frames using the loaded LLaVA-NeXT-Video model.
        """
        if not self.model or not self.processor:
            logger.error("LLaVA-NeXT-Video model or processor not loaded. Cannot generate text.")
            return None
        if not frames:
            logger.warning("No frames provided to LLaVA-NeXT-Video for text generation.")
            return None

        try:
            # Construct the prompt using LLaVA's chat template format
            # The processor.tokenizer.chat_template might be available, or use the standard format
            # Example from docs: "USER: <video>\nWhy is this video funny?\nASSISTANT:"
            # The <video> token is a special placeholder.
            # The processor should handle inserting this correctly if it's part of its vocabulary or template logic.
            # For LLaVA-NeXT, the prompt passed to processor usually contains the <video> placeholder.
            
            full_prompt = f"USER: <video>\n{prompt_instruction}\nASSISTANT:"
            
            # The LLaVA-NeXT-Video processor takes `videos` as a list of PIL images for a single video input
            inputs = self.processor(text=full_prompt, videos=frames, return_tensors="pt").to(self.device)

            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False) # Use do_sample=False for more deterministic output initially
            
            # Decode the generated IDs, skipping the prompt part
            # The processor's decode function or tokenizer.decode should handle this.
            # The generated_ids will contain both the input_ids (prompt) and the newly generated tokens.
            # We need to decode only the generated part.
            # One common way is:
            input_token_len = inputs.input_ids.shape[1]
            generated_text = self.processor.batch_decode(generated_ids[:, input_token_len:], skip_special_tokens=True)[0]

            return generated_text.strip()

        except Exception as e:
            logger.error(f"Error during LLaVA-NeXT-Video text generation: {e}", exc_info=True)
            return None

    def get_rich_description(self, frames: list[Image.Image], shot_number: int) -> str | None:
        instruction = "Describe this video clip in detail. What are the main subjects, objects, and the overall scene?"
        logger.info(f"Generating rich description for shot {shot_number} using LLaVA-NeXT-Video...")
        return self._generate_text_from_shot_frames(frames, prompt_instruction=instruction, max_new_tokens=150)

    def get_actions(self, frames: list[Image.Image], shot_number: int) -> list[str] | None:
        instruction = "List the primary actions or events occurring in this video clip. If multiple actions, separate them with commas."
        logger.info(f"Identifying actions for shot {shot_number} using LLaVA-NeXT-Video...")
        generated_text = self._generate_text_from_shot_frames(frames, prompt_instruction=instruction, max_new_tokens=75)
        if generated_text:
            actions = [action.strip() for action in generated_text.split(',') if action.strip()]
            return actions
        return None

    def get_genre_mood_cues(self, frames: list[Image.Image], shot_number: int) -> list[str] | None:
        instruction = (
            "Analyze this video clip. Describe its overall mood or atmosphere. "
            "List any visual elements (like colors, lighting), character expressions, or implied events "
            "that contribute to this mood. Also, list keywords that might suggest a potential genre. "
            "Separate all descriptors and keywords with commas."
        )
        logger.info(f"Extracting genre/mood cues for shot {shot_number} using LLaVA-NeXT-Video...")
        generated_text = self._generate_text_from_shot_frames(frames, prompt_instruction=instruction, max_new_tokens=100)
        if generated_text:
            cues = [cue.strip() for cue in generated_text.split(',') if cue.strip()]
            return cues
        return None

    def extract_metadata_for_shot(
        self, 
        original_video_path: str, 
        shot_info: dict, 
        num_frames_for_vllm: int = 8 # LLaVA-NeXT-Video might default to 8 frames for its video token
    ) -> dict | None:
        """
        Orchestrates metadata extraction for a single shot using LLaVA-NeXT-Video.
        """
        if not self.model or not self.processor:
            logger.error("LLaVA-NeXT-Video not initialized. Cannot extract metadata.")
            return {"error": "LLaVA-NeXT-Video not initialized."}

        logger.info(f"Preparing LLaVA-NeXT-Video input for shot {shot_info['shot_number']}...")
        frames_pil_list = self._prepare_shot_frames(
            original_video_path, 
            shot_info, 
            num_frames_to_sample=num_frames_for_vllm
        )

        if not frames_pil_list:
            logger.warning(f"Could not prepare frames for shot {shot_info['shot_number']}. Skipping VLLM metadata.")
            return {"error": f"Frame preparation failed for shot {shot_info['shot_number']}"}

        # To optimize, you can pass frames_pil_list once and generate all metadata types
        # if the model allows multiple queries on the same visual input without reprocessing.
        # However, the prompt is part of the input, so re-processing with different prompts is standard.

        description = self.get_rich_description(frames_pil_list, shot_info['shot_number'])
        actions = self.get_actions(frames_pil_list, shot_info['shot_number'])
        genre_mood_cues = self.get_genre_mood_cues(frames_pil_list, shot_info['shot_number'])
        
        return {
            "vllm_description": description,
            "vllm_actions": actions if actions else [],
            "vllm_genre_mood_keywords": genre_mood_cues if genre_mood_cues else [],
        }