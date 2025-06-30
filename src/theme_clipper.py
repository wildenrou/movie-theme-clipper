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
import pwd
import grp
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

# ===== NEW DIRECTORY MANAGEMENT FUNCTIONS =====

def get_nobody_uid_gid():
    """Get the UID and GID for the 'nobody' user."""
    try:
        nobody_user = pwd.getpwnam('nobody')
        nobody_group = grp.getgrnam('nobody')
        return nobody_user.pw_uid, nobody_group.gr_gid
    except KeyError:
        try:
            # Fallback to 'nogroup' if 'nobody' group doesn't exist
            nobody_user = pwd.getpwnam('nobody')
            nogroup = grp.getgrnam('nogroup')
            return nobody_user.pw_uid, nogroup.gr_gid
        except KeyError:
            logger.warning("Could not find 'nobody' user or group, using numeric fallback")
            return 65534, 65534  # Standard nobody UID/GID

def set_directory_permissions(directory_path, recursive=True):
    """Set directory ownership to 'nobody' and permissions to read/write for all."""
    try:
        if not os.path.exists(directory_path):
            logger.warning(f"Directory does not exist: {directory_path}")
            return False
        
        nobody_uid, nobody_gid = get_nobody_uid_gid()
        
        # Set permissions for the directory itself
        os.chmod(directory_path, 0o755)  # rwxr-xr-x for directories
        os.chown(directory_path, nobody_uid, nobody_gid)
        
        if recursive:
            for root, dirs, files in os.walk(directory_path):
                # Set directory permissions
                for dir_name in dirs:
                    dir_path = os.path.join(root, dir_name)
                    try:
                        os.chmod(dir_path, 0o755)
                        os.chown(dir_path, nobody_uid, nobody_gid)
                    except (OSError, PermissionError) as e:
                        logger.warning(f"Could not set permissions for directory {dir_path}: {e}")
                
                # Set file permissions
                for file_name in files:
                    file_path = os.path.join(root, file_name)
                    try:
                        os.chmod(file_path, 0o644)  # rw-r--r-- for files
                        os.chown(file_path, nobody_uid, nobody_gid)
                    except (OSError, PermissionError) as e:
                        logger.warning(f"Could not set permissions for file {file_path}: {e}")
        
        logger.info(f"Set permissions for directory: {os.path.basename(directory_path)}")
        return True
        
    except Exception as e:
        logger.error(f"Error setting permissions for {directory_path}: {e}")
        return False

def find_case_insensitive_directories(movie_folder, target_names):
    """Find directories with case-insensitive matching to target names."""
    found_dirs = {}
    
    try:
        if not os.path.exists(movie_folder) or not os.path.isdir(movie_folder):
            return found_dirs
        
        for item in os.listdir(movie_folder):
            item_path = os.path.join(movie_folder, item)
            if os.path.isdir(item_path):
                for target in target_names:
                    if item.lower() == target.lower():
                        if target not in found_dirs:
                            found_dirs[target] = []
                        found_dirs[target].append((item, item_path))
        
        return found_dirs
        
    except Exception as e:
        logger.error(f"Error scanning directory {movie_folder}: {e}")
        return {}

def merge_directories_preserving_symlinks(source_dir, target_dir):
    """Merge source directory into target directory, preserving symlinks."""
    try:
        if not os.path.exists(source_dir):
            return True
        
        if not os.path.exists(target_dir):
            os.makedirs(target_dir, exist_ok=True)
        
        merge_count = 0
        
        for item in os.listdir(source_dir):
            source_item = os.path.join(source_dir, item)
            target_item = os.path.join(target_dir, item)
            
            if os.path.islink(source_item):
                # Handle symlinks
                if os.path.exists(target_item) or os.path.islink(target_item):
                    logger.info(f"Target already exists, skipping symlink: {item}")
                    continue
                
                # Copy the symlink
                link_target = os.readlink(source_item)
                os.symlink(link_target, target_item)
                logger.info(f"Moved symlink: {item}")
                merge_count += 1
                
            elif os.path.isdir(source_item):
                # Handle subdirectories
                if not os.path.exists(target_item):
                    shutil.move(source_item, target_item)
                    logger.info(f"Moved directory: {item}")
                    merge_count += 1
                else:
                    # Recursively merge subdirectories
                    merge_directories_preserving_symlinks(source_item, target_item)
                    merge_count += 1
                    
            elif os.path.isfile(source_item):
                # Handle regular files
                if os.path.exists(target_item):
                    # Check if files are the same
                    if os.path.getsize(source_item) == os.path.getsize(target_item):
                        logger.info(f"Target file already exists with same size, skipping: {item}")
                        continue
                    else:
                        # Rename with suffix to avoid overwrite
                        base, ext = os.path.splitext(item)
                        counter = 1
                        while os.path.exists(os.path.join(target_dir, f"{base}_{counter}{ext}")):
                            counter += 1
                        new_name = f"{base}_{counter}{ext}"
                        target_item = os.path.join(target_dir, new_name)
                        logger.info(f"Renaming duplicate file: {item} -> {new_name}")
                
                shutil.move(source_item, target_item)
                logger.info(f"Moved file: {item}")
                merge_count += 1
        
        return merge_count > 0
        
    except Exception as e:
        logger.error(f"Error merging directories {source_dir} -> {target_dir}: {e}")
        return False

def check_and_merge_duplicate_directories(movie_folder):
    """Check for and merge duplicate Backdrops/Trailers directories."""
    target_names = ['Backdrops', 'Trailers']
    merged_count = 0
    
    try:
        found_dirs = find_case_insensitive_directories(movie_folder, target_names)
        
        for target_name in target_names:
            if target_name in found_dirs and len(found_dirs[target_name]) > 1:
                logger.info(f"Found {len(found_dirs[target_name])} {target_name} directories in {os.path.basename(movie_folder)}")
                
                # Sort to prioritize the correctly named directory
                dirs_list = found_dirs[target_name]
                dirs_list.sort(key=lambda x: (x[0] != target_name, x[0].lower()))
                
                primary_dir = None
                primary_path = None
                
                # Find or create the primary directory with correct case
                for dir_name, dir_path in dirs_list:
                    if dir_name == target_name:
                        primary_dir = dir_name
                        primary_path = dir_path
                        break
                
                # If no correctly named directory exists, rename the first one
                if not primary_dir:
                    old_name, old_path = dirs_list[0]
                    primary_path = os.path.join(movie_folder, target_name)
                    
                    try:
                        os.rename(old_path, primary_path)
                        logger.info(f"Renamed {old_name} -> {target_name}")
                        dirs_list[0] = (target_name, primary_path)
                        primary_dir = target_name
                    except OSError as e:
                        logger.error(f"Could not rename {old_name} to {target_name}: {e}")
                        primary_path = old_path
                        primary_dir = old_name
                
                # Merge all other directories into the primary one
                for dir_name, dir_path in dirs_list[1:]:
                    if dir_path != primary_path:
                        logger.info(f"Merging {dir_name} into {primary_dir}")
                        if merge_directories_preserving_symlinks(dir_path, primary_path):
                            try:
                                # Remove the now-empty source directory
                                if os.path.exists(dir_path) and not os.listdir(dir_path):
                                    os.rmdir(dir_path)
                                    logger.info(f"Removed empty directory: {dir_name}")
                                    merged_count += 1
                                elif os.path.exists(dir_path):
                                    logger.warning(f"Source directory not empty after merge: {dir_name}")
                            except OSError as e:
                                logger.warning(f"Could not remove directory {dir_name}: {e}")
        
        return merged_count > 0
        
    except Exception as e:
        logger.error(f"Error checking for duplicate directories in {movie_folder}: {e}")
        return False

def ensure_standard_directory_names(movie_folder):
    """Ensure Backdrops and Trailers directories use standard capitalization."""
    target_dirs = {
        'backdrops': 'Backdrops',
        'trailers': 'Trailers'
    }
    
    renamed_count = 0
    
    try:
        if not os.path.exists(movie_folder) or not os.path.isdir(movie_folder):
            return 0
        
        for item in os.listdir(movie_folder):
            item_path = os.path.join(movie_folder, item)
            if os.path.isdir(item_path):
                lower_name = item.lower()
                if lower_name in target_dirs and item != target_dirs[lower_name]:
                    new_name = target_dirs[lower_name]
                    new_path = os.path.join(movie_folder, new_name)
                    
                    if not os.path.exists(new_path):
                        try:
                            os.rename(item_path, new_path)
                            logger.info(f"Renamed directory: {item} -> {new_name}")
                            renamed_count += 1
                        except OSError as e:
                            logger.error(f"Could not rename {item} to {new_name}: {e}")
                    else:
                        logger.warning(f"Cannot rename {item} to {new_name}: target already exists")
        
        return renamed_count
        
    except Exception as e:
        logger.error(f"Error ensuring standard directory names in {movie_folder}: {e}")
        return 0

def normalize_movie_directories(movie_folder):
    """Complete directory normalization process for a movie folder."""
    try:
        movie_name = os.path.basename(movie_folder)
        logger.debug(f"Normalizing directories for: {movie_name}")
        
        # Step 1: Check and merge duplicate directories
        duplicates_merged = check_and_merge_duplicate_directories(movie_folder)
        
        # Step 2: Ensure standard naming
        renames_done = ensure_standard_directory_names(movie_folder)
        
        # Step 3: Create directories if they don't exist (with correct case)
        backdrops_dir = os.path.join(movie_folder, "Backdrops")
        trailers_dir = os.path.join(movie_folder, "Trailers")
        
        os.makedirs(backdrops_dir, exist_ok=True)
        os.makedirs(trailers_dir, exist_ok=True)
        
        # Step 4: Set proper permissions and ownership
        set_directory_permissions(backdrops_dir, recursive=True)
        set_directory_permissions(trailers_dir, recursive=True)
        
        if duplicates_merged or renames_done:
            logger.info(f"Directory normalization completed for {movie_name}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error normalizing directories for {movie_folder}: {e}")
        return False

# ===== ORIGINAL AND ENHANCED TRAILER MANAGEMENT FUNCTIONS =====

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
    
    # Use the normalized directory structure
    trailers_dir = os.path.join(movie_folder_path, "Trailers")
    backdrops_dir = os.path.join(movie_folder_path, "Backdrops")
    
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
    """Main processing function with enhanced trailer management and directory normalization."""
    
    logger.info(f"🎬 Movie Theme Clipper starting...")
    logger.info(f"📁 Movie path: {base_path}")
    logger.info(f"⏱️ Clip length: {clip_length}s")
    logger.info(f"🎭 Method: {method}")
    logger.info(f"🚀 GPU acceleration: {use_gpu}")
    logger.info(f"🎞️ Process trailers: {process_trailers}")
    logger.info(f"📂 Directory normalization: enabled")
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
    directories_normalized = 0

    for movie_folder_path, movie_file_path in tqdm(movies_data, desc="Processing movies", unit="movie"):
        movie_name = os.path.basename(movie_file_path)
        movie_folder = os.path.dirname(movie_file_path)
        
        # NEW: Normalize directory structure first
        if normalize_movie_directories(movie_folder):
            directories_normalized += 1
        
        # Use normalized directory paths
        backdrop_dir = os.path.join(movie_folder, "Backdrops")
        theme_clip_output_path = os.path.join(backdrop_dir, "theme.mp4")

        # Process trailers
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
    logger.info(f"   Directories normalized: {directories_normalized}")

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
