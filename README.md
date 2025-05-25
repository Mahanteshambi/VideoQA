# Video Analysis Suite

This project aims to build a comprehensive suite of tools for advanced video understanding, enabling semantic search, content-based analysis, and interactive chat functionalities with video content.

Currently, the suite includes modules for video ingestion and temporal segmentation.

## Current Capabilities

* **Module 1: Video Ingestion & Preprocessing (`src/video_ingestion`)**
    * Extracts frames from videos at a configurable rate (using MoviePy).
    * Demuxes and saves the full audio track (using ffmpeg-python, typically to WAV).
    * Transcribes audio to text using OpenAI Whisper, providing both detailed JSON (with timestamps) and plain text outputs.
    * Collects technical metadata (resolution, duration, FPS, codecs, etc.) using ffprobe.
    * Organizes all outputs into a structured directory for each processed video.

* **Module 2: Temporal Video Segmentation (`src/scene_segmentation`)**
    * **Shot Detection:** Identifies shot boundaries using PySceneDetect (ContentDetector).
    * **Multimodal Feature Extraction (Per Shot):**
        * **Visual:** Extracts semantic embeddings from keyframes using OpenAI CLIP (ViT-B/32). Supports configurable number of keyframes per shot (averaged).
        * **Textual:** Generates embeddings for transcript segments corresponding to each shot using Sentence Transformers (e.g., `all-MiniLM-L6-v2`).
        * **Audio:** Extracts basic features (e.g., mean RMS energy via Librosa) as a placeholder for more advanced audio embeddings.
    * **Scene Grouping:** Groups detected shots into semantically cohesive scenes using a weighted late fusion of similarity scores (cosine similarity) from visual, audio, and textual modalities. Scene breaks are determined by a configurable similarity threshold.
    * Outputs a hierarchical structure of scenes, each containing its constituent shots and associated features.

## Prerequisites

* Python 3.8 or higher.
* **FFmpeg:** Must be installed on your system and accessible in the system's PATH. Download from [ffmpeg.org](https://ffmpeg.org/download.html).
* (Optional) An NVIDIA GPU with CUDA installed for significantly faster processing with PyTorch-based models (CLIP, Whisper, Sentence Transformers).

## Setup and Installation

1.  **Clone the repository** (if applicable, or ensure you have the project files).
2.  **Install UV** (if you haven't already): Follow instructions at [astral.sh/uv](https://astral.sh/uv).
3.  **Navigate to the project root directory:**
    ```bash
    cd video_analysis_project
    ```
4.  **Create and activate a virtual environment:**
    ```bash
    uv venv
    source .venv/bin/activate  # For Linux/macOS
    # On Windows: .venv\Scripts\activate
    ```
5.  **Install dependencies:**
    ```bash
    uv pip install -e .
    ```
    This command installs the project in editable mode along with all dependencies specified in `pyproject.toml`.

## Running the Demos

Demo scripts are provided in the `scripts/` directory to showcase the functionality of each module. Ensure your virtual environment is activated.

* **Video Ingestion Demo:**
    Processes a sample video (a dummy video will be created if one doesn't exist) and saves outputs to `processed_videos_output_module2_1/`.
    ```bash
    python scripts/run_ingestion_demo.py
    ```

* **Scene Segmentation Demo:**
    First, it ensures the necessary video ingestion outputs exist (running ingestion if needed), then processes the video for shot detection and scene grouping. Outputs are saved to `processed_videos_output_module2_2_scenes/`.
    ```bash
    python scripts/run_scene_segmentation_demo.py
    ```

## Key Configuration

The scene segmentation quality is highly dependent on parameters found in `scripts/run_scene_segmentation_demo.py` when calling `segment_video_into_scenes`. Key parameters to experiment with include:

* `num_keyframes_per_shot`: Number of frames averaged for a shot's visual feature. (Recommended: 3-5)
* `scene_similarity_threshold`: Threshold (0-1) for scene breaks. Lower values lead to longer, fewer scenes.
* `modality_weights`: Dictionary defining the influence of visual, audio, and textual similarities.
* `shot_detector_threshold` (for PySceneDetect): Controls shot detection sensitivity.

## Current Status

* Initial implementation of video ingestion and hierarchical (shot-to-scene) temporal segmentation modules.
* Core feature extraction using CLIP (visual) and Sentence Transformers (textual) is in place.
* Audio feature extraction is currently basic (placeholder) and needs enhancement.
* Parameter tuning for scene segmentation is an ongoing process.

## Next Steps / Future Work

* Integrate advanced audio embedding models (e.g., VGGish, YAMNet).
* Experiment with alternative shot detectors (e.g., TransNetV2).
* Develop more sophisticated methods for modeling intra-shot temporal dynamics.
* Build downstream modules for content analysis (genre/mood detection, object/action recognition) using the segmented scenes and shots.
* Develop the search and chat interface.

## License

This project is licensed under the MIT License (as specified in `pyproject.toml`).
