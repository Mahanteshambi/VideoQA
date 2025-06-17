from pathlib import Path
from videoqa.json_processing import process_all_shot_features

def main():
    # Get the project root directory (2 levels up from this file)
    project_root = Path(__file__).parent.parent.parent
    process_all_shot_features(base_dir=project_root)

if __name__ == "__main__":
    main() 