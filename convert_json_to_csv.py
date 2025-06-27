#!/usr/bin/env python3
"""
Utility script to convert existing JSON shot features to CSV format.
This is useful for converting existing pipeline outputs to the new CSV format.
"""

import json
import csv
import sys
from pathlib import Path
from typing import Dict, List, Any

def convert_json_to_csv(json_file: Path, csv_file: Path) -> None:
    """
    Convert a JSON file containing shot features to CSV format.
    
    Args:
        json_file: Path to the JSON file containing shot features
        csv_file: Path to output CSV file
    """
    
    # Read JSON data
    with open(json_file, 'r', encoding='utf-8') as f:
        shots_data = json.load(f)
    
    # Define CSV fieldnames
    fieldnames = [
        'shot_number', 'start_time_seconds', 'end_time_seconds', 
        'start_frame', 'end_frame', 'duration_seconds',
        'metadata_ShotDescription', 'metadata_GenreCues', 'metadata_SubgenreCues',
        'metadata_AdjectiveTheme', 'metadata_Mood', 'metadata_SettingContext',
        'metadata_ContentDescriptors', 'metadata_LocationHints_Regional',
        'metadata_LocationHints_International', 'metadata_SearchKeywords'
    ]
    
    # Write CSV
    with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for shot in shots_data:
            row = {
                'shot_number': shot.get('shot_number', ''),
                'start_time_seconds': shot.get('start_time_seconds', ''),
                'end_time_seconds': shot.get('end_time_seconds', ''),
                'start_frame': shot.get('start_frame', ''),
                'end_frame': shot.get('end_frame', ''),
                'duration_seconds': shot.get('duration_seconds', ''),
            }
            
            # Extract metadata fields
            vllm_metadata = shot.get('vllm_metadata', {})
            if isinstance(vllm_metadata, str):
                # If metadata is a string (raw response), try to parse it
                try:
                    vllm_metadata = json.loads(vllm_metadata)
                except:
                    vllm_metadata = {}
            
            # Map metadata fields to CSV columns
            metadata_mapping = {
                'metadata_ShotDescription': vllm_metadata.get('ShotDescription', ''),
                'metadata_GenreCues': json.dumps(vllm_metadata.get('GenreCues', []), ensure_ascii=False),
                'metadata_SubgenreCues': json.dumps(vllm_metadata.get('SubgenreCues', []), ensure_ascii=False),
                'metadata_AdjectiveTheme': json.dumps(vllm_metadata.get('AdjectiveTheme', []), ensure_ascii=False),
                'metadata_Mood': json.dumps(vllm_metadata.get('Mood', []), ensure_ascii=False),
                'metadata_SettingContext': json.dumps(vllm_metadata.get('SettingContext', []), ensure_ascii=False),
                'metadata_ContentDescriptors': json.dumps(vllm_metadata.get('ContentDescriptors', []), ensure_ascii=False),
                'metadata_LocationHints_Regional': json.dumps(vllm_metadata.get('LocationHints_Regional', []), ensure_ascii=False),
                'metadata_LocationHints_International': json.dumps(vllm_metadata.get('LocationHints_International', []), ensure_ascii=False),
                'metadata_SearchKeywords': json.dumps(vllm_metadata.get('SearchKeywords', []), ensure_ascii=False),
            }
            
            row.update(metadata_mapping)
            writer.writerow(row)
    
    print(f"✅ Converted {json_file} to {csv_file}")
    print(f"   Processed {len(shots_data)} shots")

def main():
    """Main function to handle command line arguments."""
    
    if len(sys.argv) != 3:
        print("Usage: python convert_json_to_csv.py <input_json_file> <output_csv_file>")
        print("Example: python convert_json_to_csv.py Hair_Love_smolvlm_shot_features.json Hair_Love_shot_metadata.csv")
        sys.exit(1)
    
    json_file = Path(sys.argv[1])
    csv_file = Path(sys.argv[2])
    
    if not json_file.exists():
        print(f"Error: JSON file not found: {json_file}")
        sys.exit(1)
    
    try:
        convert_json_to_csv(json_file, csv_file)
    except Exception as e:
        print(f"Error converting file: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 