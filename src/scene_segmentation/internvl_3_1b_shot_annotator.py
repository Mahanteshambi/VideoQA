# video_analysis_project/src/scene_segmentation/internvl_3_1b_shot_annotator.py

import torch
from PIL import Image
import numpy as np
from pathlib import Path
import logging
import moviepy.editor as mp
from transformers import AutoModelForCausalLM, AutoTokenizer
from torchvision import transforms

logger = logging.getLogger(__name__)

# --- Device Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"InternVL3_1B_ShotAnnotator: Using device: {DEVICE}")

def build_transform(input_size=448):
    """Build transformation pipeline for images."""
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform

def dynamic_preprocess(img, image_size=448, use_thumbnail=True, max_num=4):
    """Preprocess image with dynamic tiling."""
    w, h = img.size
    scale = image_size / min(w, h)
    w, h = int(w * scale), int(h * scale)
    img = img.resize((w, h), Image.Resampling.LANCZOS)
    
    if use_thumbnail and (w > image_size or h > image_size):
        num_patch_w = min(max_num, (w + image_size - 1) // image_size)
        num_patch_h = min(max_num, (h + image_size - 1) // image_size)
        patches = []
        for i in range(num_patch_h):
            for j in range(num_patch_w):
                x1, y1 = j * image_size, i * image_size
                x2, y2 = min(x1 + image_size, w), min(y1 + image_size, h)
                patch = img.crop((x1, y1, x2, y2))
                if x2 - x1 != image_size or y2 - y1 != image_size:
                    patch = patch.resize((image_size, image_size), Image.Resampling.LANCZOS)
                patches.append(patch)
        return patches
    return [img]

def get_frame_indices(start_time, end_time, fps, num_segments=32):
    """Get frame timepoints for video sampling."""
    if num_segments == 1:
        return [start_time + (end_time - start_time) / 2]
    
    return np.linspace(start_time, end_time, num_segments, endpoint=True)

class InternVL3_1B_ShotAnnotator:
    def __init__(self, model_checkpoint: str = "OpenGVLab/InternVL3-1B",
                 device: str = DEVICE):
        """
        Initializes the InternVL 3-1B annotator.
        Args:
            model_checkpoint (str): Hugging Face model identifier
            device (str): "cuda" or "cpu"
        """
        self.device = device
        self.model = None
        self.tokenizer = None
        self.model_checkpoint = model_checkpoint
        
        try:
            logger.info(f"Loading InternVL tokenizer from: {self.model_checkpoint}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_checkpoint, 
                trust_remote_code=True
            )
            
            logger.info(f"Loading InternVL model from: {self.model_checkpoint}")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_checkpoint,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            ).to(self.device)
            self.model.eval()
            
            logger.info(f"InternVL model '{self.model_checkpoint}' loaded successfully on {self.device}.")

        except Exception as e:
            logger.error(f"Failed to load InternVL model or tokenizer from '{self.model_checkpoint}': {e}", exc_info=True)
            raise

    def _process_video_frames(
        self,
        video_path: str,
        start_time: float,
        end_time: float,
        num_segments: int = 8,
        input_size: int = 448,
        max_num: int = 1
    ) -> tuple[torch.Tensor, list[int]]:
        """Process video frames using InternVL's approach with moviepy."""
        video_clip = None
        try:
            video_clip = mp.VideoFileClip(video_path)
            fps = video_clip.fps
            
            # Get frame timepoints
            timepoints = get_frame_indices(start_time, end_time, fps, num_segments)
            
            pixel_values_list, num_patches_list = [], []
            transform = build_transform(input_size=input_size)
            
            for t in timepoints:
                # Extract frame
                frame_np = video_clip.get_frame(min(t, video_clip.duration - 0.001))
                img = Image.fromarray(frame_np).convert('RGB')
                
                # Process frame
                img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
                pixel_values = [transform(tile) for tile in img]
                pixel_values = torch.stack(pixel_values)
                
                num_patches_list.append(pixel_values.shape[0])
                pixel_values_list.append(pixel_values)
            
            pixel_values = torch.cat(pixel_values_list)
            # Convert to float16 to match model dtype
            pixel_values = pixel_values.to(dtype=torch.float16, device=self.device)
            return pixel_values, num_patches_list
            
        except Exception as e:
            logger.error(f"Error processing video frames: {e}", exc_info=True)
            raise
        finally:
            if video_clip:
                video_clip.close()

    def _generate_text_with_internvl_chat(
        self, 
        video_path: str,
        start_time: float,
        end_time: float,
        instruction: str,
        num_segments: int = 8,
        max_new_tokens: int = 150
    ) -> str | None:
        if not self.model or not self.tokenizer:
            logger.error("InternVL model or tokenizer not loaded.")
            return None

        try:
            # Process video frames
            pixel_values, num_patches_list = self._process_video_frames(
                video_path,
                start_time,
                end_time,
                num_segments=num_segments
            )
            
            # Create video frame prefix
            video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
            question = video_prefix + instruction

            # Generation config
            generation_config = dict(max_new_tokens=1024, do_sample=True)

            # Generate response
            response, _ = self.model.chat(
                self.tokenizer,
                pixel_values,
                question,
                generation_config,
                num_patches_list=num_patches_list,
                history=None,
                return_history=True
            )
            
            return response.strip()

        except Exception as e:
            logger.error(f"Error during InternVL text generation: {e}", exc_info=True)
            return None

    def get_rich_description(self, video_path: str, shot_info: dict) -> str | None:
        instruction = "Describe this video segment in detail. What are the main subjects, objects, and the overall scene?"
        logger.info(f"Generating rich description for shot {shot_info['shot_number']} using InternVL...")
        return self._generate_text_with_internvl_chat(
            video_path,
            shot_info["start_time_seconds"],
            shot_info["end_time_seconds"],
            instruction
        )

    def get_actions(self, video_path: str, shot_info: dict) -> list[str] | None:
        instruction = "List the primary actions or events occurring in this video segment. If multiple actions, separate them with commas."
        logger.info(f"Identifying actions for shot {shot_info['shot_number']} using InternVL...")
        generated_text = self._generate_text_with_internvl_chat(
            video_path,
            shot_info["start_time_seconds"],
            shot_info["end_time_seconds"],
            instruction
        )
        if generated_text:
            actions = [action.strip() for action in generated_text.split(',') if action.strip().lower() not in ["none", "n/a", ""]]
            return actions if actions else None
        return None

    def get_genre_mood_cues(self, video_path: str, shot_info: dict) -> list[str] | None:
        instruction = (
            "Analyze this video segment. Describe its overall mood or atmosphere. "
            "List any visual elements (colors, lighting), character expressions, or implied events "
            "that contribute to this mood. Also, list keywords that might suggest a potential genre. "
            "Separate all descriptors and keywords with commas."
        )
        logger.info(f"Extracting genre/mood cues for shot {shot_info['shot_number']} using InternVL...")
        generated_text = self._generate_text_with_internvl_chat(
            video_path,
            shot_info["start_time_seconds"],
            shot_info["end_time_seconds"],
            instruction
        )
        if generated_text:
            cues = [cue.strip() for cue in generated_text.split(',') if cue.strip().lower() not in ["none", "n/a", ""]]
            return cues if cues else None
        return None

    def extract_metadata_for_shot(
        self, 
        original_video_path: str, 
        shot_info: dict,
        num_keyframes_to_sample: int = 8  # Changed to match InternVL's default
    ) -> dict | None:
        if not self.model or not self.tokenizer:
            return {"error": "InternVL model not initialized."}

        logger.info(f"Preparing InternVL input for shot {shot_info['shot_number']}...")
        
        # Extract metadata using video processing
        description = self.get_rich_description(original_video_path, shot_info)
        actions = self.get_actions(original_video_path, shot_info)
        genre_mood_cues = self.get_genre_mood_cues(original_video_path, shot_info)
        
        return {
            "vllm_description": description,
            "vllm_actions": actions if actions else [],
            "vllm_genre_mood_keywords": genre_mood_cues if genre_mood_cues else [],
        }