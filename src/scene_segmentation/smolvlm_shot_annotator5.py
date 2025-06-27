import cv2
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import logging
import uuid
import moviepy.editor as mp
import traceback
from pathlib import Path
logger = logging.getLogger(__name__)

class SmolVLMShotAnnotatorV5:

    def __init__(self) -> None:
        # Choose one:
        model_id = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"  # or -2.2B-Instruct, -256M-Video-Instruct, -500M-Video-Instruct

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(model_id,
                                                        torch_dtype=torch.bfloat16)#.to("cuda")
        self.UNIFIED_JSON_EXTRACTION_PROMPT_INSTRUCTION = """You are a scene understanding assistant. Analyze the provided video shot carefully. Extract and return metadata solely based on the visual and implied content of this shot.

            Respond with one valid, complete, and properly formatted JSON object that strictly follows the schema below. Do not include any explanations or comments. Only output the filled JSON object.

            ⛔ Do not repeat or describe the schema itself.
            ✅ Do include specific values based on your analysis of the video content.

            Ensure your JSON response:
            - Has all required keys as per schema (do not rename fields)
            - Uses properly quoted strings and comma-separated lists
            - Closes all arrays and objects correctly
            - Fills all values realistically based on the video

            🎯 Fill the following schema:

            {
            "ShotDescription": "...",
            "GenreCues": [
                {
                "genre_hint": "...",
                "prominence_in_shot": ...
                }
            ],
            "SubgenreCues": ["..."],
            "AdjectiveTheme": ["..."],
            "Mood": ["..."],
            "SettingContext": ["..."],
            "ContentDescriptors": ["..."],
            "LocationHints_Regional": ["..."],
            "LocationHints_International": ["..."],
            "SearchKeywords": ["..."]
            }

"""

    def sample_video_frames(self, video_path, num_frames=8):
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = [int(i * total / num_frames) for i in range(num_frames)]
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret: break
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(img))
        cap.release()
        return frames
    
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
# frames = sample_video_frames("sample_videos/Hair Love.mp4", num_frames=8)
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
        inputs = self.processor.apply_chat_template(messages,
                                       add_generation_prompt=True,
                                       tokenize=True,
                                       return_dict=True,
                                       return_tensors="pt").to(self.model.device)
        # If on GPU, ensure BF16 dtype
        for k,v in inputs.items():
            if torch.is_floating_point(v):
                inputs[k] = v.to(torch.bfloat16)

        outputs = self.model.generate(**inputs, max_new_tokens=1024, do_sample=False)
        decoded = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        if "<|assistant|>" in decoded:
            response = decoded.split("<|assistant|>")[-1].strip()
        elif "Assistant:" in decoded:
            response = decoded.split("Assistant:")[-1].strip()
        else:
            response = decoded.strip()
        # print("---- RAW DECODED OUTPUT ----")
        # print(decoded)
        print("---- FINAL PARSED RESPONSE ----")
        print(response)
        # print("Generated description:", response)
        return response

# messages = [
#     {"role": "user", "content": [
#         {"type": "video", "path": "sample_videos/Hair Love.mp4"},
#         {"type": "text", "text": "Describe this video in detail."}
#     ]}
# ]





