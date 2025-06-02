import json
import numpy as np
from typing import Dict, List, Any
from collections import Counter
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def load_json_file(file_path: str) -> List[Dict[str, Any]]:
    """Load and parse a JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def calculate_field_presence(annotations: List[Dict[str, Any]], field: str) -> float:
    """Calculate the percentage of shots where a field has non-empty values."""
    total = len(annotations)
    non_empty = sum(1 for shot in annotations 
                   if shot.get('vllm_metadata', {}).get('vllm_generated_json_metadata', {}).get(field))
    return (non_empty / total) * 100 if total > 0 else 0

def analyze_mood_distribution(annotations: List[Dict[str, Any]]) -> Counter:
    """Analyze the distribution of moods across all shots."""
    moods = []
    for shot in annotations:
        metadata = shot.get('vllm_metadata', {}).get('vllm_generated_json_metadata', {})
        if 'Mood' in metadata:
            moods.extend(metadata['Mood'])
    return Counter(moods)

def analyze_genre_distribution(annotations: List[Dict[str, Any]]) -> Dict[str, float]:
    """Analyze the distribution and prominence of genres."""
    genres = {}
    for shot in annotations:
        metadata = shot.get('vllm_metadata', {}).get('vllm_generated_json_metadata', {})
        if 'GenreCues' in metadata:
            for cue in metadata['GenreCues']:
                if isinstance(cue, dict) and 'genre_hint' in cue and 'prominence_in_shot' in cue:
                    genre = cue['genre_hint']
                    prominence = float(cue['prominence_in_shot'])
                    if genre not in genres:
                        genres[genre] = []
                    genres[genre].append(prominence)
    
    # Calculate average prominence per genre
    return {genre: np.mean(prominences) for genre, prominences in genres.items()}

def calculate_description_stats(annotations: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate statistics about shot descriptions."""
    lengths = []
    for shot in annotations:
        metadata = shot.get('vllm_metadata', {}).get('vllm_generated_json_metadata', {})
        if 'ShotDescription' in metadata:
            lengths.append(len(metadata['ShotDescription'].split()))
    
    return {
        'avg_length': np.mean(lengths) if lengths else 0,
        'min_length': min(lengths) if lengths else 0,
        'max_length': max(lengths) if lengths else 0,
        'std_length': np.std(lengths) if lengths else 0
    }

def compare_shot_descriptions(llava_data: List[Dict[str, Any]], 
                            internvl_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compare shot descriptions between the two models."""
    comparisons = []
    
    for llava_shot, internvl_shot in zip(llava_data, internvl_data):
        llava_metadata = llava_shot.get('vllm_metadata', {}).get('vllm_generated_json_metadata', {})
        internvl_metadata = internvl_shot.get('vllm_metadata', {}).get('vllm_generated_json_metadata', {})
        
        shot_comparison = {
            'shot_number': llava_shot['shot_number'],
            'llava_description': llava_metadata.get('ShotDescription', ''),
            'internvl_description': internvl_metadata.get('ShotDescription', ''),
            'llava_mood': llava_metadata.get('Mood', []),
            'internvl_mood': internvl_metadata.get('Mood', []),
            'llava_genres': [g.get('genre_hint') for g in llava_metadata.get('GenreCues', [])],
            'internvl_genres': [g.get('genre_hint') for g in internvl_metadata.get('GenreCues', [])]
        }
        comparisons.append(shot_comparison)
    
    return comparisons

def analyze_vllm_outputs(llava_file: str, internvl_file: str) -> Dict[str, Any]:
    """Main function to analyze and compare VLLM outputs."""
    logger.info(f"Analyzing VLLM outputs from {llava_file} and {internvl_file}")
    
    # Load the data
    llava_data = load_json_file(llava_file)
    internvl_data = load_json_file(internvl_file)
    
    # Ensure both files have the same number of shots
    if len(llava_data) != len(internvl_data):
        logger.warning(f"Number of shots differs: LLaVA ({len(llava_data)}) vs InternVL ({len(internvl_data)})")
    
    analysis = {
        'num_shots': {
            'llava': len(llava_data),
            'internvl': len(internvl_data)
        },
        
        'field_presence': {
            'llava': {
                field: calculate_field_presence(llava_data, field)
                for field in ['ShotDescription', 'GenreCues', 'Mood', 'SettingContext', 'ContentDescriptors']
            },
            'internvl': {
                field: calculate_field_presence(internvl_data, field)
                for field in ['ShotDescription', 'GenreCues', 'Mood', 'SettingContext', 'ContentDescriptors']
            }
        },
        
        'mood_distribution': {
            'llava': dict(analyze_mood_distribution(llava_data)),
            'internvl': dict(analyze_mood_distribution(internvl_data))
        },
        
        'genre_distribution': {
            'llava': analyze_genre_distribution(llava_data),
            'internvl': analyze_genre_distribution(internvl_data)
        },
        
        'description_stats': {
            'llava': calculate_description_stats(llava_data),
            'internvl': calculate_description_stats(internvl_data)
        },
        
        'shot_comparisons': compare_shot_descriptions(llava_data, internvl_data)
    }
    
    return analysis

def print_analysis_report(analysis: Dict[str, Any]) -> None:
    """Print a formatted report of the analysis."""
    print("\n=== VLLM Output Analysis Report ===\n")
    
    print("Shot Counts:")
    print(f"LLaVA: {analysis['num_shots']['llava']} shots")
    print(f"InternVL: {analysis['num_shots']['internvl']} shots\n")
    
    print("Field Presence (% of shots with non-empty values):")
    for field in analysis['field_presence']['llava'].keys():
        print(f"{field}:")
        print(f"  LLaVA: {analysis['field_presence']['llava'][field]:.1f}%")
        print(f"  InternVL: {analysis['field_presence']['internvl'][field]:.1f}%")
    print()
    
    print("Top 5 Most Common Moods:")
    print("LLaVA:")
    for mood, count in sorted(analysis['mood_distribution']['llava'].items(), 
                            key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {mood}: {count}")
    print("InternVL:")
    for mood, count in sorted(analysis['mood_distribution']['internvl'].items(), 
                            key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {mood}: {count}")
    print()
    
    print("Genre Analysis (Average Prominence):")
    print("LLaVA:")
    for genre, prominence in sorted(analysis['genre_distribution']['llava'].items(), 
                                  key=lambda x: x[1], reverse=True):
        print(f"  {genre}: {prominence:.1f}")
    print("InternVL:")
    for genre, prominence in sorted(analysis['genre_distribution']['internvl'].items(), 
                                  key=lambda x: x[1], reverse=True):
        print(f"  {genre}: {prominence:.1f}")
    print()
    
    print("Description Statistics:")
    for model in ['llava', 'internvl']:
        print(f"{model.upper()}:")
        stats = analysis['description_stats'][model]
        print(f"  Average length: {stats['avg_length']:.1f} words")
        print(f"  Min length: {stats['min_length']} words")
        print(f"  Max length: {stats['max_length']} words")
        print(f"  Standard deviation: {stats['std_length']:.1f} words")
    print()
    
    print("Shot-by-Shot Comparison Examples (first 3 shots):")
    for comp in analysis['shot_comparisons'][:3]:
        print(f"\nShot {comp['shot_number']}:")
        print("LLaVA Description:", comp['llava_description'])
        print("InternVL Description:", comp['internvl_description'])
        print("LLaVA Mood:", ', '.join(comp['llava_mood']))
        print("InternVL Mood:", ', '.join(comp['internvl_mood']))
        print("LLaVA Genres:", ', '.join(comp['llava_genres']))
        print("InternVL Genres:", ', '.join(comp['internvl_genres']))

if __name__ == "__main__":
    # Example usage
    llava_file = "processed_videos_output_module2_2_scenes/Hair Love/Hair Love_llava_next_shot_features.json"
    internvl_file = "processed_videos_output_module2_2_scenes/Hair Love/Hair Love_internvl_3_1b_shot_features.json"
    
    analysis_results = analyze_vllm_outputs(llava_file, internvl_file)
    print_analysis_report(analysis_results) 