import os
import random
import numpy as np
import librosa
import subprocess
import logging
import sys
import json
import hashlib
import time
import shutil
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

# ===== TRAILER MANAGEMENT FUNCTIONS =====

def get_file_hash(file_path):
    """Generate a hash of the file for tracking processed files."""
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        logger.warning(f"Could not generate hash for {file_path}: {e}")
        return None

def load_processed_files(tracking_file):
    """Load the list of already processed files."""
    if tracking_file and os.path.exists(tracking_file):
        try:
            with open(tracking_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load tracking file: {e}")
    return {}

def save_processed_files(tracking_file, processed_files):
    """Save the list of processed files."""
    if not tracking_file:
        return
    try:
        os.makedirs(os.path.dirname(tracking_file), exist_ok=True)
        with open(tracking_file, 'w') as f:
            json.dump(processed_files, f, indent=2)
    except Exception as e:
        logger.warning(f"Could not save tracking file: {e}")

def is_file_already_processed(file_path, processed_files, volume_factor):
    """Check if a file has already been processed with the same volume factor."""
    file_hash = get_file_hash(file_path)
    if not file_hash:
        return False
    
    if file_hash in processed_files:
        stored_volume = processed_files[file_hash].get('volume_factor')
        if stored_volume == volume_factor:
            logger.debug(f"File already processed with volume factor {volume_factor}: {os.path.basename(file_path)}")
            return True
    
    return False

def mark_file_as_processed(file_path, processed_files, volume_factor):
    """Mark a file as processed with the given volume factor."""
    file_hash = get_file_hash(file_path)
    if file_hash:
        processed_files[file_hash] = {
            'file_path': file_path,
            'volume_factor': volume_factor,
            'processed_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'file_size': os.path.getsize(file_path)
        }

def clean_invalid_symlinks(directory):
    """Remove invalid symbolic links in the specified directory."""
    cleaned_count = 0
    
    try:
        if not os.path.exists(directory) or not os.path.isdir(directory):
            return 0
        
        for filename in os.listdir(directory):
            full_path = os.path.join(directory, filename)
            if os.path.islink(full_path):
                target_path = os.readlink(full_path)
                if not os.path.exists(os.path.abspath(os.path.join(directory, target_path))):
                    logger.info(f"Removing invalid symlink: {filename} -> {target_path}")
                    try:
                        os.remove(full_path)
                        cleaned_count += 1
                    except OSError as e:
                        logger.warning(f"Failed to remove invalid symlink {filename}: {e}")
        
        return cleaned_count
        
    except Exception as e:
        logger.error(f"Error in clean_invalid_symlinks: {e}")
        return 0

def find_and_fix_trailer(trailers_dir):
    """Find misnamed trailer files and rename them to trailer.mp4."""
    try:
        if not os.path.exists(trailers_dir) or not os.path.isdir(trailers_dir):
            return None
        
        for filename in os.listdir(trailers_dir):
            full_path = os.path.join(trailers_dir, filename)
            
            if not os.path.isfile(full_path):
                continue
                
            name, ext = os.path.splitext(filename)
            if name.lower().endswith("trailer") and ext.lower() == ".mp4":
                target_path = os.path.join(trailers_dir, "trailer.mp4")
                logger.info(f"Renaming misnamed trailer: {filename} -> trailer.mp4")
                try:
                    os.rename(full_path, target_path)
                    return target_path
                except OSError as e:
                    logger.error(f"Failed to rename trailer {filename}: {e}")
                    return None
        
        return None
        
    except Exception as e:
        logger.error(f"Error in find_and_fix_trailer: {e}")
        return None

def reduce_video_volume_internal(video_path, volume_factor=0.5, processed_files=None):
    """Reduce the volume of a video file using internal FFmpeg."""
    if not os.path.isfile(video_path):
        logger.error(f"Video file not found: {video_path}")
        return False
    
    # Check if file has already been processed
    if processed_files and is_file_already_processed(video_path, processed_files, volume_factor):
        return True
    
    logger.info(f"Reducing volume by {int((1-volume_factor)*100)}% for {os.path.basename(video_path)}")
    
    # Create temp file with .mp4 extension so FFmpeg recognizes the format
    temp_output = video_path + ".volume_temp.mp4"
    
    try:
        if os.path.exists(temp_output):
            os.remove(temp_output)
        
        # FFmpeg command for volume reduction with explicit format
        cmd = [
            'ffmpeg', '-y', '-v', 'warning',
            '-i', video_path,
            '-c:v', 'copy',  # Copy video without re-encoding
            '-af', f'volume={volume_factor}',  # Reduce audio volume
            '-c:a', 'aac',   # Re-encode audio
            '-movflags', '+faststart',
            '-f', 'mp4',     # Explicitly specify MP4 format
            temp_output
        ]
        
        logger.debug(f"Volume reduction command: {' '.join(cmd[:-1])} [output_file]")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode != 0:
            logger.error(f"FFmpeg volume reduction failed for {os.path.basename(video_path)}")
            if result.stderr:
                # Show key error lines
                stderr_lines = result.stderr.strip().split('\n')
                error_lines = [line for line in stderr_lines if any(keyword in line.lower() 
                              for keyword in ['error', 'failed', 'unable', 'invalid'])]
                if error_lines:
                    logger.error(f"FFmpeg errors: {'; '.join(error_lines[-2:])}")
                else:
                    logger.error(f"FFmpeg stderr (last 200 chars): {result.stderr[-200:]}")
            return False
        
        # Check output file
        if not os.path.exists(temp_output):
            logger.error("Volume reduction output file was not created")
            return False
        
        output_size = os.path.getsize(temp_output)
        input_size = os.path.getsize(video_path)
        
        if output_size < input_size * 0.1:
            logger.error("Volume reduction output file suspiciously small")
            os.remove(temp_output)
            return False
        
        # Replace original with processed version
        original_stat = os.stat(video_path)
        
        # Create backup name for safety
        backup_path = video_path + ".original_backup"
        
        # Move original to backup, then move processed to original location
        shutil.move(video_path, backup_path)
        shutil.move(temp_output, video_path)
        
        # Restore original file permissions
        os.chmod(video_path, original_stat.st_mode)
        
        # Remove backup if everything succeeded
        os.remove(backup_path)
        
        # Mark as processed
        if processed_files is not None:
            mark_file_as_processed(video_path, processed_files, volume_factor)
        
        logger.info(f"Volume reduction successful for {os.path.basename(video_path)}")
        return True
        
    except subprocess.TimeoutExpired:
        logger.error(f"Volume reduction timeout for {os.path.basename(video_path)}")
        return False
    except Exception as e:
        logger.error(f"Volume reduction failed for {os.path.basename(video_path)}: {e}")
        return False
    finally:
        # Clean up temporary files
        for temp_file in [temp_output, video_path + ".original_backup"]:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass

def manage_movie_trailers(movie_folder_path, process_volume=False, volume_factor=0.5, processed_files=None):
    """Manage trailers for a single movie folder."""
    movie_name = os.path.basename(movie_folder_path)
    trailers_dir = os.path.join(movie_folder_path, "Trailers")
    backdrops_dir = os.path.join(movie_folder_path, "Backdrops")
    
    # Create directories if they don't exist
    os.makedirs(trailers_dir, exist_ok=True)
    os.makedirs(backdrops_dir, exist_ok=True)
    
    # Clean invalid symlinks
    cleaned = clean_invalid_symlinks(trailers_dir) + clean_invalid_symlinks(backdrops_dir)
    if cleaned > 0:
        logger.info(f"Cleaned {cleaned} invalid symlinks in {movie_name}")
    
    trailer_file = os.path.join(trailers_dir, "trailer.mp4")
    symlink_file = os.path.join(backdrops_dir, "theme2.mp4")
    relative_target = os.path.relpath(trailer_file, backdrops_dir)
    
    # Ensure trailer file exists
    if not os.path.isfile(trailer_file):
        fixed_path = find_and_fix_trailer(trailers_dir)
        if not fixed_path:
            logger.debug(f"No trailer file found for {movie_name}")
            return False
    
    # Process audio volume if requested
    if process_volume:
        reduce_video_volume_internal(trailer_file, volume_factor, processed_files)
    
    # Manage symlink
    if os.path.islink(symlink_file):
        existing_target = os.readlink(symlink_file)
        if os.path.abspath(os.path.join(backdrops_dir, existing_target)) == os.path.abspath(trailer_file):
            logger.debug(f"Symlink already correct for {movie_name}")
            return True
        else:
            logger.info(f"Fixing incorrect symlink for {movie_name}")
            os.remove(symlink_file)
    elif os.path.exists(symlink_file):
        logger.warning(f"theme2.mp4 is a real file in {movie_name}, skipping symlink creation")
        return False
    
    try:
        os.symlink(relative_target, symlink_file)
        logger.info(f"Created symlink theme2.mp4 -> trailer.mp4 for {movie_name}")
        return True
    except Exception as e:
        logger.error(f"Failed to create symlink for {movie_name}: {e}")
        return False

# ===== ORIGINAL THEME CLIPPER FUNCTIONS =====

def check_gpu_availability():
    """Check if Intel GPU is available and working."""
    try:
        if not os.path.exists('/dev/dri/renderD128'):
            logger.warning("Intel GPU device /dev/dri/renderD128 not found")
            return False
            
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
    """Extracts a theme clip using ffmpeg with HDR/high bit-depth compatibility."""
    try:
        # Build FFmpeg command with HDR handling
        final_ffmpeg_cmd = ["ffmpeg", "-y", "-v", "info"]
        final_ffmpeg_cmd.extend(["-analyzeduration", "100M", "-probesize", "100M"])
        final_ffmpeg_cmd.extend(["-ss", str(start_time), "-i", movie_path, "-t", str(clip_length)])

        # Handle HDR/high bit-depth content by converting pixel format first
        video_filters = []
        
        # First convert to standard 8-bit format, then scale
        video_filters.append("format=yuv420p")  # Convert to 8-bit first
        video_filters.append("scale=1280:720:flags=lanczos")  # Then scale with specific algorithm
        
        final_ffmpeg_cmd.extend(["-vf", ",".join(video_filters)])

        # Use software encoding for better compatibility
        final_ffmpeg_cmd.extend(["-c:v", "libx264", "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p"])

        # Audio encoding - handle various input formats
        final_ffmpeg_cmd.extend(["-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2"])
        
        # Ensure MP4 output
        final_ffmpeg_cmd.extend(["-f", "mp4"])
        final_ffmpeg_cmd.append(output_path)

        # Execute FFmpeg
        logger.debug(f"FFmpeg command: {' '.join(final_ffmpeg_cmd)}")
        
        process = subprocess.run(final_ffmpeg_cmd, capture_output=True, text=True, 
                               encoding='utf-8', errors='ignore', timeout=600)

        if process.returncode != 0:
            logger.error(f"FFmpeg failed for {os.path.basename(movie_path)}")
            if process.stderr:
                # Show key error lines
                stderr_lines = process.stderr.strip().split('\n')
                error_lines = [line for line in stderr_lines if any(keyword in line.lower() 
                              for keyword in ['error', 'failed', 'impossible', 'unsupported'])]
                if error_lines:
                    logger.error(f"FFmpeg errors: {'; '.join(error_lines[-2:])}")
                else:
                    logger.error(f"FFmpeg stderr (last 300 chars): {process.stderr[-300:]}")
            return False
            
        return True

    except subprocess.TimeoutExpired:
        logger.error(f"FFmpeg timeout for {os.path.basename(movie_path)}")
        return False
    except Exception as e:
        logger.error(f"Exception during clip extraction for {os.path.basename(movie_path)}: {e}")
        return False

def process_movies(base_path, clip_length=18, method='visual', start_buffer=120, 
                  end_ignore_pct=0.3, force=False, use_gpu=False, 
                  process_trailers=True, process_trailer_volume=False, trailer_volume_factor=0.5):
    """Main processing function with enhanced trailer management."""
    
    logger.info(f"🎬 Movie Theme Clipper starting...")
    logger.info(f"📁 Movie path: {base_path}")
    logger.info(f"⏱️ Clip length: {clip_length}s")
    logger.info(f"🎭 Method: {method}")
    logger.info(f"🚀 GPU acceleration: {use_gpu}")
    logger.info(f"🎞️ Process trailers: {process_trailers}")
    if process_trailers and process_trailer_volume:
        logger.info(f"🔊 Trailer volume reduction: {int((1-trailer_volume_factor)*100)}%")
    
    movies_data = find_movie_files(base_path)
    if not movies_data:
        logger.error("No movies found to process")
        return

    # Setup trailer tracking
    processed_trailers = {}
    tracking_file = None
    
    if process_trailers and process_trailer_volume:
        tracking_file = "/logs/processed_trailers.json"
        processed_trailers = load_processed_files(tracking_file)
        logger.info(f"Loaded {len(processed_trailers)} previously processed trailers")

    processed = 0
    skipped = 0
    failed = 0
    trailers_processed = 0
    trailers_skipped = 0

    for movie_folder_path, movie_file_path in tqdm(movies_data, desc="Processing movies", unit="movie"):
        movie_name = os.path.basename(movie_file_path)
        movie_folder = os.path.dirname(movie_file_path)
        backdrop_dir = os.path.join(movie_folder, "Backdrops")
        theme_clip_output_path = os.path.join(backdrop_dir, "theme.mp4")

        # Process trailers first
        if process_trailers:
            trailer_result = manage_movie_trailers(
                movie_folder, 
                process_trailer_volume, 
                trailer_volume_factor, 
                processed_trailers
            )
            if trailer_result:
                trailers_processed += 1
            else:
                trailers_skipped += 1

        # Skip theme creation if already exists
        if os.path.exists(theme_clip_output_path) and os.path.getsize(theme_clip_output_path) > 0 and not force:
            logger.info(f"⏩ Skipping {os.path.basename(movie_folder)} - theme.mp4 already exists")
            skipped += 1
            continue

        os.makedirs(backdrop_dir, exist_ok=True)
        logger.info(f"🎞️ Processing theme clip: {os.path.basename(movie_folder)}")

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

            # Calculate clip timing
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
                logger.info(f"✅ Successfully created theme clip for {os.path.basename(movie_folder)}")
                processed += 1
            else:
                logger.error(f"❌ Failed to create theme clip for {os.path.basename(movie_folder)}")
                failed += 1

        except subprocess.TimeoutExpired:
            logger.error(f"Timeout processing {movie_name}")
            failed += 1
        except Exception as e:
            logger.error(f"Error processing {movie_name}: {e}")
            failed += 1

    # Save trailer tracking data
    if process_trailers and process_trailer_volume and tracking_file:
        save_processed_files(tracking_file, processed_trailers)

    logger.info(f"🎉 Processing complete:")
    logger.info(f"   Theme clips: {processed} processed, {skipped} skipped, {failed} failed")
    if process_trailers:
        logger.info(f"   Trailers: {trailers_processed} processed, {trailers_skipped} skipped")

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
    
    # New trailer management options
    process_trailers = get_env_bool('PROCESS_TRAILERS', True)
    process_trailer_volume = get_env_bool('PROCESS_TRAILER_VOLUME', False)
    trailer_volume_factor = get_env_float('TRAILER_VOLUME_FACTOR', 0.5)

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
    logger.info(f"  Process trailers: {process_trailers}")
    logger.info(f"  Process trailer volume: {process_trailer_volume}")
    if process_trailer_volume:
        logger.info(f"  Trailer volume factor: {trailer_volume_factor}")

    try:
        process_movies(
            base_path=movie_library_path,
            clip_length=clip_length,
            method=method,
            start_buffer=start_buffer,
            end_ignore_pct=end_ignore_pct,
            force=force,
            use_gpu=use_gpu,
            process_trailers=process_trailers,
            process_trailer_volume=process_trailer_volume,
            trailer_volume_factor=trailer_volume_factor
        )
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)