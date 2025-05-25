import os
import sys
print("Current PATH:", os.environ.get("PATH"))
print("Current Python path:", sys.path)

# Add ffmpeg path
os.environ["PATH"] = "/opt/homebrew/bin:" + os.environ.get("PATH", "")

try:
    import ffmpeg
    print("Successfully imported ffmpeg")
    print("ffmpeg.__file__:", ffmpeg.__file__)
except ImportError as e:
    print("Failed to import ffmpeg:", e)
    print("Detailed error information:", str(e)) 