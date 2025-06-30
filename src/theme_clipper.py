import os
import random
import numpy as np
import librosa
import subprocess
import logging
import sys
from moviepy import VideoFileClip
import moviepy

from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

import contextlib

# Setup logging
def setup_logging():
    log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('/logs/theme_clipper.log') if os.path.exists('/logs') else logging.NullHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# Color Output for console
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"

@contextlib.contextmanager
def suppress_output():
    """Suppresses stdout and stderr."""
    with open(os.devnull, 'w') as devnull_file:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull_file
        sys.stderr = devnull_file
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

def check_gpu_availability():
    """Check if Intel GPU is available and working."""
    try:
        # Check if device exists
        if not os.path.exists('/dev/dri/renderD128'):
            logger.warning("Intel GPU device /dev/dri/renderD128 not found")
            return False
            
        # Test VAAPI
        result = subprocess.run(['vainfo'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and 'H264' in result.stdout:
            logger.info("Intel GPU with H.264 VAAPI support detected")
            return True
        else:
            logger.warning("Intel GPU detected but H.264 VAAPI not available")
            return False
            
    except Exception as e:
        logger.warning(f"GPU check failed: {e}")
        return False

def find_movie_files(base_path, extensions=None):
    """Finds movie files in subdirectories of base_path."""
    if extensions is None:
        extensions = ('.mp4', '.mkv', '.mov', '.avi', '.flv', '.wmv', '.webm', '.m4v')

    skip_keywords = ['trailer', 'featurette', 'behindthescenes', 'sample', 'bts', 
                     'deletedscenes', 'extra', 'short', 'preview']

    movie_paths = []
    logger.info(f"Searching for movie files in: {base_path}")
    
    if not os.path.isdir(base_path):
        logger.error(f"Base path '{base_path}' not found or is not a directory")
        return movie_paths

    try:
        for foldername in os.listdir(base_path):
            full_folder_path = os.path.join(base_path, foldername)
            if not os.path.isdir(full_folder_path):
                continue

            video_candidates = []
            for item_name in os.listdir(full_folder_path):
                lower_item_name = item_name.lower()
                if not any(k in lower_item_name for k in skip_keywords) and lower_item_name.endswith(extensions):
                    full_item_path = os.path.join(full_folder_path, item_name)
                    try:
                        if os.path.isfile(full_item_path):
                            size = os.path.getsize(full_item_path)
                            video_candidates.append((size, full_item_path))
                    except OSError as e:
                        logger.warning(f"Could not access {full_item_path}: {e}")

            if video_candidates:
                video_candidates.sort(key=lambda x: x[0], reverse=True)
                largest_file_path = video_candidates[0][1]
                movie_paths.append((full_folder_path, largest_file_path))

    except Exception as e:
        logger.error(f"Error scanning movie directories: {e}")

    if not movie_paths:
        logger.warning(f"No movie files found in subdirectories of '{base_path}'")
    else:
        logger.info(f"Found {len(movie_paths)} movie(s) to process")
    
    return movie_paths

def get_random_clip_start(duration, clip_length, start_buffer, end_ignore_pct):
    """Calculates a random start time for a clip."""
    end_ignore_pct = max(0, min(1, end_ignore_pct))
    latest_valid_start_time = duration * (1 - end_ignore_pct) - clip_length
    effective_start_buffer = min(start_buffer, latest_valid_start_time)

    if effective_start_buffer >= latest_valid_start_time:
        if duration <= clip_length: 
            return 0
        return max(0, start_buffer) if start_buffer < duration - clip_length else 0

    return random.uniform(effective_start_buffer, latest_valid_start_time)

def get_env_bool(env_var, default=False):
    """Convert environment variable to boolean."""
    value = os.getenv(env_var, str(default)).lower()
    return value in ('true', '1', 'yes', 'on')

def get_env_float(env_var, default=0.0):
    """Convert environment variable to float."""
    try:
        return float(os.getenv(env_var, str(default)))
    except ValueError:
        return default

def get_env_int(env_var, default=0):
    """Convert environment variable to int."""
    try:
        return int(os.getenv(env_var, str(default)))
    except ValueError:
        return default

if __name__ == '__main__':
    # Get configuration from environment variables
    movie_library_path = os.getenv('MOVIE_PATH', '/movies')
    clip_length = get_env_int('CLIP_LENGTH', 18)
    method = os.getenv('METHOD', 'visual')
    start_buffer = get_env_int('START_BUFFER', 120)
    end_ignore_pct = get_env_float('END_IGNORE_PCT', 0.3)
    use_gpu = get_env_bool('USE_GPU', True)
    force = get_env_bool('FORCE', False)

    # Validate GPU if requested
    if use_gpu:
        gpu_available = check_gpu_availability()
        if not gpu_available:
            logger.warning("GPU requested but not available, falling back to CPU")
            use_gpu = False

    logger.info(f"Starting with configuration:")
    logger.info(f"  Movie path: {movie_library_path}")
    logger.info(f"  Clip length: {clip_length}s")
    logger.info(f"  Method: {method}")
    logger.info(f"  GPU acceleration: {use_gpu}")
    logger.info(f"  Force overwrite: {force}")

    # This is a simplified version for setup testing
    print("🎬 Movie Theme Clipper setup complete!")
    print("This container is ready to process your movie collection.")
    print("For full functionality, use the complete application code.")