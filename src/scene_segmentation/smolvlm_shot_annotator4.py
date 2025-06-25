from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
import logging
import moviepy.editor as mp
import traceback
import uuid
from pathlib import Path
logger = logging.getLogger(__name__)

class SmolVLMShotAnnotatorV4:
    def __init__(self, model_path: str = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"):
        self.model_path = model_path
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            # _attn_implementation="flash_attention_2"
        )#.to("cuda")
        self.UNIFIED_JSON_EXTRACTION_PROMPT_INSTRUCTION = """Analyze the provided video shot carefully. Your task is to extract comprehensive metadata based SOLELY on the visual and implied content of THIS SHOT.
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

    def extract_metadata_for_shot(self, original_video_path: str, shot_info: dict, num_keyframes_to_sample: int = 4) -> dict:

        logger.info(f"Creating temporary video for shot {shot_info['shot_number']}...")
        temp_video_path = self._create_temp_shot_video(original_video_path, shot_info)
        if not temp_video_path:
            return {"error": "Failed to create temporary video for shot"}
    
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "path": temp_video_path},
                    {"type": "text", "text": self.UNIFIED_JSON_EXTRACTION_PROMPT_INSTRUCTION}
                ]
            }]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device, dtype=torch.bfloat16)
        generated_ids = self.model.generate(**inputs, do_sample=False)
        generated_texts = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )
        return generated_texts[0]

if __name__ == "__main__":
    annotator = SmolVLMShotAnnotatorV4()
    print(annotator.extract_metadata_for_shot("sample_videos/Hair Love.mp4", {"shot_number": 1}))

# model_path = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
# processor = AutoProcessor.from_pretrained(model_path)
# model = AutoModelForImageTextToText.from_pretrained(
#     model_path,
#     torch_dtype=torch.bfloat16,
#     _attn_implementation="flash_attention_2"
# ).to("cuda")

# messages = [
#     {
#         "role": "user",
#         "content": [
#             {"type": "video", "path": "path_to_video.mp4"},
#             {"type": "text", "text": "Describe this video in detail"}
#         ]
#     },
# ]

# inputs = processor.apply_chat_template(
#     messages,
#     add_generation_prompt=True,
#     tokenize=True,
#     return_dict=True,
#     return_tensors="pt",
# ).to(model.device, dtype=torch.bfloat16)

# generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=64)
# generated_texts = processor.batch_decode(
#     generated_ids,
#     skip_special_tokens=True,
# )

# print(generated_texts[0])