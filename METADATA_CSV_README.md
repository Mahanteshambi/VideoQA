# Metadata CSV Output Feature

This document describes the new metadata CSV output functionality added to the VideoQA scene segmentation pipeline.

## Overview

The pipeline now generates a CSV file containing detailed metadata for each shot, including:
- Basic shot information (timing, frames, duration)
- AI-generated metadata from VLLM models (descriptions, genres, themes, etc.)

## CSV Output Format

The CSV file contains the following columns:

| Column | Description |
|--------|-------------|
| `shot_number` | Sequential shot number |
| `start_time_seconds` | Start time of the shot in seconds |
| `end_time_seconds` | End time of the shot in seconds |
| `start_frame` | Starting frame number |
| `end_frame` | Ending frame number |
| `duration_seconds` | Duration of the shot in seconds |
| `metadata_ShotDescription` | AI-generated description of the shot content |
| `metadata_GenreCues` | Genre indicators with prominence levels (JSON array) |
| `metadata_SubgenreCues` | Subgenre elements visible in the shot (JSON array) |
| `metadata_AdjectiveTheme` | Descriptive adjectives capturing the shot's theme (JSON array) |
| `metadata_Mood` | Emotional atmosphere and mood conveyed (JSON array) |
| `metadata_SettingContext` | Environmental and contextual details (JSON array) |
| `metadata_ContentDescriptors` | Specific content elements, objects, actions (JSON array) |
| `metadata_LocationHints_Regional` | Regional or local location indicators (JSON array) |
| `metadata_LocationHints_International` | International or global location indicators (JSON array) |
| `metadata_SearchKeywords` | Relevant keywords for content search (JSON array) |

## Files Modified

### 1. `src/scene_segmentation/smolvlm_shot_annotator5.py`

**Changes made:**
- Enhanced the prompt to generate more detailed and structured metadata
- Added JSON parsing functionality with error handling
- Improved metadata field extraction and validation
- Added cleanup of temporary video files

**Key improvements:**
- More specific prompt instructions for each metadata field
- Robust JSON parsing with fallback handling
- Better error reporting and logging

### 2. `src/scene_segmentation/pipeline.py`

**Changes made:**
- Added `dump_shot_metadata_to_csv()` function
- Integrated CSV output into the main pipeline
- Added CSV file path to pipeline outputs
- Enhanced error handling for CSV generation

**New functionality:**
- Automatic CSV generation after feature extraction
- Proper handling of JSON arrays in CSV format
- UTF-8 encoding support for international characters

## Usage

### Running the Updated Pipeline

The pipeline automatically generates CSV output when run:

```python
from src.scene_segmentation.pipeline import segment_video_into_scenes

results = segment_video_into_scenes(
    video_path="sample_videos/Hair Love.mp4",
    ingestion_output_dir="processed_videos_output",
    scene_segmentation_output_dir="processed_videos_output_module2_2_scenes",
    vllm_annotator_type="smolvlm"
)
```

### Output Files

The pipeline now generates:
- `{video_name}_shot_metadata.csv` - Tabular metadata for all shots
- `{video_name}_{annotator_type}_shot_features.json` - Original JSON format (preserved)
- Other existing output files

### Converting Existing Data

Use the utility script to convert existing JSON outputs to CSV:

```bash
python convert_json_to_csv.py Hair_Love_smolvlm_shot_features.json Hair_Love_shot_metadata.csv
```

## Testing

Run the test script to verify the functionality:

```bash
python test_metadata_pipeline.py
```

This will:
1. Run the pipeline with metadata extraction
2. Generate CSV output
3. Display the first few lines of the CSV file
4. Show pipeline status and any errors

## Example CSV Output

```csv
shot_number,start_time_seconds,end_time_seconds,start_frame,end_frame,duration_seconds,metadata_ShotDescription,metadata_GenreCues,metadata_SubgenreCues,metadata_AdjectiveTheme,metadata_Mood,metadata_SettingContext,metadata_ContentDescriptors,metadata_LocationHints_Regional,metadata_LocationHints_International,metadata_SearchKeywords
1,0.0,2.5,0,75,2.5,"A young girl with natural hair sitting in a chair looking at herself in a mirror","[{\"genre_hint\": \"animation\", \"prominence_in_shot\": \"high\"}]","[\"family\", \"slice of life\"]","[\"intimate\", \"reflective\", \"personal\"]","[\"contemplative\", \"quiet\", \"focused\"]","[\"bedroom\", \"mirror\", \"personal space\"]","[\"girl\", \"hair\", \"mirror\", \"chair\", \"reflection\"]","[\"home\", \"bedroom\"]","[\"universal\", \"everyday\"]","[\"hair care\", \"self reflection\", \"mirror\", \"girl\", \"natural hair\"]"
```

## Notes

- JSON arrays are stored as JSON strings in CSV cells for compatibility
- The pipeline maintains backward compatibility with existing JSON outputs
- Error handling ensures the pipeline continues even if metadata extraction fails for some shots
- Temporary video files are automatically cleaned up after processing

## Troubleshooting

### Common Issues

1. **CSV file not generated**: Check if the pipeline completed successfully and look for errors in the logs
2. **Empty metadata fields**: This may indicate issues with the VLLM model or video processing
3. **JSON parsing errors**: The pipeline includes fallback handling for malformed JSON responses

### Debug Mode

Enable debug logging to see detailed information about metadata extraction:

```python
import logging
logging.getLogger('scene_segmentation').setLevel(logging.DEBUG)
``` 