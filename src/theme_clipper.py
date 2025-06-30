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

# Setup logging with better error handling
def setup_logging():
    log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    
    # Start with console handler
    handlers = [logging.StreamHandler(sys.stdout)]
    
    # Try to add file handler if possible
    try:
        if os.path.exists('/logs') and os.access('/logs', os.W_OK):
            handlers.append(logging.FileHandler('/logs/theme_clipper.log'))
        elif os.path.exists('/logs'):
            print("Warning: /logs directory exists but is not writable. Logging to console only.")
    except Exception as e:
        print(f"Warning: Could not set up file logging: {e}. Logging to console only.")
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format=log_format,
        handlers=handlers
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

def extract_theme_clip(movie_path, output_path, start_time, clip_length, use_gpu=False):
    """Extracts a theme clip using ffmpeg with optional GPU acceleration."""
    try:
        # Build FFmpeg command with better error handling
        final_ffmpeg_cmd = ["ffmpeg", "-y", "-v", "info"]  # Changed from "warning" to "info" for better debugging
        final_ffmpeg_cmd.extend(["-analyzeduration", "100M", "-probesize", "100M"])  # Increased for complex files
        final_ffmpeg_cmd.extend(["-ss", str(start_time), "-i", movie_path, "-t", str(clip_length)])

        # Handle complex video formats more robustly
        video_filters = []
        
        # Scale down and ensure compatibility
        video_filters.append("scale=1280:720:force_original_aspect_ratio=decrease")
        video_filters.append("pad=1280:720:(ow-iw)/2:(oh-ih)/2")  # Add padding if needed
        
        if video_filters:
            final_ffmpeg_cmd.extend(["-vf", ",".join(video_filters)])

        # Use software encoding for better compatibility with complex formats
        if use_gpu:
            # Try GPU first, but with fallback
            try:
                # Test if GPU encoding works
                test_cmd = ["ffmpeg", "-f", "lavfi", "-i", "testsrc=duration=1:size=320x240:rate=1", 
                           "-c:v", "h264_vaapi", "-vaapi_device", "/dev/dri/renderD128", "-f", "null", "-"]
                test_result = subprocess.run(test_cmd, capture_output=True, timeout=5)
                if test_result.returncode == 0:
                    final_ffmpeg_cmd.extend(["-c:v", "h264_vaapi", "-vaapi_device", "/dev/dri/renderD128", "-qp", "23"])
                else:
                    raise Exception("GPU test failed")
            except:
                # Fallback to CPU
                logger.warning(f"GPU encoding failed for {os.path.basename(movie_path)}, using CPU")
                final_ffmpeg_cmd.extend(["-c:v", "libx264", "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p"])
        else:
            final_ffmpeg_cmd.extend(["-c:v", "libx264", "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p"])

        # Audio encoding - handle various input formats
        final_ffmpeg_cmd.extend(["-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2"])
        
        # Force output format
        final_ffmpeg_cmd.extend(["-f", "mp4"])
        final_ffmpeg_cmd.append(output_path)

        # Execute FFmpeg
        logger.debug(f"FFmpeg command: {' '.join(final_ffmpeg_cmd)}")
        
        process = subprocess.run(final_ffmpeg_cmd, capture_output=True, text=True, 
                               encoding='utf-8', errors='ignore', timeout=600)  # Increased timeout

        if process.returncode != 0:
            logger.error(f"FFmpeg failed for {os.path.basename(movie_path)}")
            logger.error(f"FFmpeg command: {' '.join(final_ffmpeg_cmd)}")
            if process.stderr:
                # Show actual FFmpeg error
                stderr_lines = process.stderr.strip().split('\n')
                error_lines = [line for line in stderr_lines if 'error' in line.lower() or 'failed' in line.lower()]
                if error_lines:
                    logger.error(f"FFmpeg errors: {'; '.join(error_lines[-3:])}")  # Last 3 error lines
                else:
                    logger.error(f"FFmpeg stderr (last 500 chars): {process.stderr[-500:]}")
            return False
            
        return True

    except subprocess.TimeoutExpired:
        logger.error(f"FFmpeg timeout for {os.path.basename(movie_path)}")
        return False
    except Exception as e:
        logger.error(f"Exception during clip extraction for {os.path.basename(movie_path)}: {e}")
        return False

def process_movies(base_path, clip_length=18, method='visual', start_buffer=120, 
                  end_ignore_pct=0.3, force=False, use_gpu=False):
    """Main processing function."""
    
    logger.info(f"🎬 Movie Theme Clipper starting...")
    logger.info(f"📁 Movie path: {base_path}")
    logger.info(f"⏱️ Clip length: {clip_length}s")
    logger.info(f"🎭 Method: {method}")
    logger.info(f"🚀 GPU acceleration: {use_gpu}")
    
    movies_data = find_movie_files(base_path)
    if not movies_data:
        logger.error("No movies found to process")
        return

    processed = 0
    skipped = 0
    failed = 0

    for movie_folder_path, movie_file_path in tqdm(movies_data, desc="Processing movies", unit="movie"):
        movie_name = os.path.basename(movie_file_path)
        backdrop_dir = os.path.join(movie_folder_path, "Backdrops")
        theme_clip_output_path = os.path.join(backdrop_dir, "theme.mp4")

        # Skip if theme already exists
        if os.path.exists(theme_clip_output_path) and os.path.getsize(theme_clip_output_path) > 0 and not force:
            logger.info(f"⏩ Skipping {movie_name} - theme.mp4 already exists")
            skipped += 1
            continue

        os.makedirs(backdrop_dir, exist_ok=True)
        logger.info(f"🎞️ Processing: {movie_name}")

        try:
            # Get movie duration using ffprobe
            probe_cmd = [
                "ffprobe", "-v", "error", "-analyzeduration", "10M", "-probesize", "10M",
                "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
                movie_file_path
            ]
            duration_process = subprocess.run(probe_cmd, capture_output=True, text=True, 
                                           check=True, encoding='utf-8', errors='ignore', timeout=30)
            duration_str = duration_process.stdout.strip().split('\n')[0]

            if not duration_str or not duration_str.replace('.', '', 1).isdigit():
                logger.warning(f"Could not determine duration for {movie_name}")
                failed += 1
                continue
                
            duration = float(duration_str)

            if duration <= 0:
                logger.warning(f"Invalid duration for {movie_name}: {duration}")
                failed += 1
                continue

            # For now, use random selection (simplified version)
            # You can add the full analysis methods later
            current_clip_length = min(clip_length, duration * 0.8)
            if current_clip_length < 5:
                current_clip_length = min(duration, clip_length)
                start_time_sec = 0
            else:
                start_time_sec = get_random_clip_start(duration, current_clip_length, start_buffer, end_ignore_pct)

            start_time_sec = max(0, float(start_time_sec))
            if start_time_sec + current_clip_length > duration:
                start_time_sec = max(0, duration - current_clip_length)
                
            if current_clip_length <= 0:
                logger.warning(f"Invalid clip length for {movie_name}")
                failed += 1
                continue

            logger.info(f"Extracting clip: {start_time_sec:.2f}s-{start_time_sec + current_clip_length:.2f}s")
            
            # Extract the clip
            success = extract_theme_clip(movie_file_path, theme_clip_output_path, 
                                       start_time_sec, current_clip_length, use_gpu=use_gpu)

            if success and os.path.exists(theme_clip_output_path) and os.path.getsize(theme_clip_output_path) > 100:
                logger.info(f"✅ Successfully created theme clip for {movie_name}")
                processed += 1
            else:
                logger.error(f"❌ Failed to create theme clip for {movie_name}")
                failed += 1

        except subprocess.TimeoutExpired:
            logger.error(f"Timeout processing {movie_name}")
            failed += 1
        except Exception as e:
            logger.error(f"Error processing {movie_name}: {e}")
            failed += 1

    logger.info(f"🎉 Processing complete: {processed} processed, {skipped} skipped, {failed} failed")

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
    use_gpu = get_env_bool('USE_GPU', False)
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

    try:
        process_movies(
            base_path=movie_library_path,
            clip_length=clip_length,
            method=method,
            start_buffer=start_buffer,
            end_ignore_pct=end_ignore_pct,
            force=force,
            use_gpu=use_gpu
        )
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)