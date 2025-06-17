import json
import pandas as pd
from pathlib import Path
import glob
import os

def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            # Convert list to string representation
            items.append((new_key, ', '.join(map(str, v))))
        else:
            items.append((new_key, v))
    return dict(items)

def process_json_to_excel(input_json_path, output_excel_path):
    # Read JSON file
    with open(input_json_path, 'r') as f:
        data = json.load(f)
    
    # Flatten each shot's data
    flattened_data = []
    for shot in data:
        # Create a copy of the shot data
        shot_data = shot.copy()
        
        # Handle the vllm_metadata separately
        if 'vllm_metadata' in shot_data:
            vllm_data = shot_data.pop('vllm_metadata')
            if 'vllm_generated_json_metadata' in vllm_data:
                metadata = vllm_data['vllm_generated_json_metadata']
                # Add each metadata field directly to shot_data with prefix
                for key, value in metadata.items():
                    shot_data[f'metadata_{key}'] = value
        
        # Flatten any remaining nested structures
        flat_shot = flatten_dict(shot_data)
        flattened_data.append(flat_shot)
    
    # Convert to DataFrame
    df = pd.DataFrame(flattened_data)
    
    # Save to Excel
    df.to_excel(output_excel_path, index=False)
    print(f"Excel file created successfully at: {output_excel_path}")

def process_all_shot_features():
    # Get the current directory
    current_dir = Path.cwd()
    
    # Input directory
    input_dir = current_dir / "processed_videos_output_module2_2_scenes"
    
    # Create output directory for Excel files if it doesn't exist
    output_dir = current_dir / "excel_output_module2_2_scenes"
    output_dir.mkdir(exist_ok=True)
    
    # Find all JSON files with "_shot_features" in their name
    json_pattern = str(input_dir / "**" / "*_shot_features.json")
    json_files = glob.glob(json_pattern, recursive=True)
    
    print(f"Found {len(json_files)} shot feature JSON files to process")
    
    # Process each JSON file
    for json_file in json_files:
        json_path = Path(json_file)
        # Create Excel filename based on JSON filename
        excel_filename = json_path.stem + ".xlsx"
        excel_path = output_dir / excel_filename
        
        print(f"\nProcessing: {json_path.name}")
        try:
            process_json_to_excel(json_file, excel_path)
        except Exception as e:
            print(f"Error processing {json_path.name}: {str(e)}")

if __name__ == "__main__":
    process_all_shot_features() 