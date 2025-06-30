
import os
import random
import subprocess
import logging
import sys
import json
import hashlib
import time
import shutil
import pwd
import grp
import warnings
import contextlib
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Setup logging
def setup_logging():
    log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    handlers = [logging.StreamHandler(sys.stdout)]
    logging.basicConfig(level=getattr(logging, log_level, logging.INFO), format=log_format, handlers=handlers)
    return logging.getLogger(__name__)

logger = setup_logging()

@contextlib.contextmanager
def suppress_output():
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

def set_directory_permissions(directory_path, recursive=True, change_owner=False):
    try:
        if not os.path.exists(directory_path):
            logger.warning(f"Directory does not exist: {directory_path}")
            return False
        
        if change_owner:
            try:
                nobody_uid, nobody_gid = pwd.getpwnam('nobody').pw_uid, grp.getgrnam('nobody').gr_gid
            except KeyError:
                nobody_uid, nobody_gid = os.getuid(), os.getgid()
        
        os.chmod(directory_path, 0o755)
        if change_owner:
            os.chown(directory_path, nobody_uid, nobody_gid)
        
        if recursive:
            for root, dirs, files in os.walk(directory_path):
                for d in dirs:
                    p = os.path.join(root, d)
                    os.chmod(p, 0o755)
                    if change_owner:
                        os.chown(p, nobody_uid, nobody_gid)
                for f in files:
                    p = os.path.join(root, f)
                    os.chmod(p, 0o644)
                    if change_owner:
                        os.chown(p, nobody_uid, nobody_gid)
        
        logger.info(f"Set permissions for directory: {os.path.basename(directory_path)}")
        return True

    except Exception as e:
        logger.error(f"Error setting permissions for {directory_path}: {e}")
        return False

def check_gpu_availability():
    try:
        result = subprocess.run(['vainfo'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and ('H264' in result.stdout or 'AVC' in result.stdout):
            logger.info("✅ Intel GPU VAAPI detected and available")
            return "vaapi"
        # Check for QSV by probing FFmpeg encoders
        ffmpeg_encoders = subprocess.run(['ffmpeg', '-hide_banner', '-encoders'], capture_output=True, text=True, timeout=10)
        if 'h264_qsv' in ffmpeg_encoders.stdout:
            logger.info("✅ Intel Quick Sync (QSV) detected and available")
            return "qsv"
    except Exception as e:
        logger.warning(f"GPU check failed: {e}")
    logger.info("⚠️ No Intel GPU acceleration detected, using CPU")
    return None

def extract_theme_clip(movie_path, output_path, start_time, clip_length, gpu_mode=None):
    try:
        if not os.access(os.path.dirname(output_path), os.W_OK):
            logger.error(f"Cannot write to directory: {os.path.dirname(output_path)}")
            return False

        cmd = ["ffmpeg", "-y", "-v", "info", "-analyzeduration", "100M", "-probesize", "100M", "-ss", str(start_time), "-i", movie_path, "-t", str(clip_length)]
        vf = "format=yuv420p,scale=1280:720:flags=lanczos"

        if gpu_mode == "vaapi":
            cmd += ["-vf", vf, "-c:v", "h264_vaapi", "-qp", "24"]
        elif gpu_mode == "qsv":
            cmd += ["-vf", vf, "-c:v", "h264_qsv", "-preset", "fast"]
        else:
            cmd += ["-vf", vf, "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p"]

        cmd += ["-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2", "-f", "mp4", output_path]

        logger.debug(f"FFmpeg command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            logger.error(f"FFmpeg failed: {result.stderr[-300:]}")
            return False
        return True

    except subprocess.TimeoutExpired:
        logger.error(f"FFmpeg timeout for {os.path.basename(movie_path)}")
        return False
    except Exception as e:
        logger.error(f"Exception during clip extraction: {e}")
        return False

def normalize_movie_directories(movie_folder):
    try:
        backdrops_dir = os.path.join(movie_folder, "Backdrops")
        trailers_dir = os.path.join(movie_folder, "Trailers")
        os.makedirs(backdrops_dir, exist_ok=True)
        os.makedirs(trailers_dir, exist_ok=True)
        set_directory_permissions(backdrops_dir, recursive=True, change_owner=False)
        set_directory_permissions(trailers_dir, recursive=True, change_owner=False)
        return True
    except Exception as e:
        logger.error(f"Error normalizing directories for {movie_folder}: {e}")
        return False

def process_movies(base_path, clip_length=18, start_buffer=120, end_ignore_pct=0.3, force=False, use_gpu=True):
    logger.info("🎬 Starting movie theme clipper")

    gpu_mode = check_gpu_availability() if use_gpu else None

    if not os.path.isdir(base_path):
        logger.error(f"Base path '{base_path}' not found or not a directory")
        return

    movie_dirs = [os.path.join(base_path, d) for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]

    for movie_folder in tqdm(movie_dirs, desc="Processing movies", unit="movie"):
        movie_files = [f for f in os.listdir(movie_folder) if f.lower().endswith(('.mp4', '.mkv', '.avi', '.mov'))]
        if not movie_files:
            logger.warning(f"No movie files in {movie_folder}")
            continue

        movie_file = max(movie_files, key=lambda f: os.path.getsize(os.path.join(movie_folder, f)))
        movie_path = os.path.join(movie_folder, movie_file)
        backdrop_dir = os.path.join(movie_folder, "Backdrops")
        theme_clip_output_path = os.path.join(backdrop_dir, "theme.mp4")

        normalize_movie_directories(movie_folder)

        if os.path.exists(theme_clip_output_path) and not force:
            logger.info(f"⏩ Skipping {os.path.basename(movie_folder)} — theme.mp4 exists")
            continue

        probe_cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", movie_path]
        try:
            result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=30)
            duration = float(result.stdout.strip())
        except Exception as e:
            logger.error(f"Could not get duration: {e}")
            continue

        if duration <= 0:
            logger.warning(f"Invalid duration for {movie_file}")
            continue

        clip_len = min(clip_length, duration * 0.8)
        if clip_len < 5:
            clip_len = min(duration, clip_length)
            start_sec = 0
        else:
            latest_start = duration * (1 - end_ignore_pct) - clip_len
            effective_buffer = min(start_buffer, latest_start)
            start_sec = random.uniform(effective_buffer, latest_start)

        logger.info(f"Extracting clip: {start_sec:.2f}s-{start_sec + clip_len:.2f}s")

        success = extract_theme_clip(movie_path, theme_clip_output_path, start_sec, clip_len, gpu_mode=gpu_mode)
        if success:
            logger.info(f"✅ Theme clip created: {os.path.basename(movie_folder)}")
        else:
            logger.error(f"❌ Failed for {os.path.basename(movie_folder)}")

if __name__ == '__main__':
    movie_library_path = os.getenv('MOVIE_PATH', '/movies')
    clip_length = int(os.getenv('CLIP_LENGTH', 18))
    start_buffer = int(os.getenv('START_BUFFER', 120))
    end_ignore_pct = float(os.getenv('END_IGNORE_PCT', 0.3))
    force = os.getenv('FORCE', 'False').lower() in ('true', '1', 'yes', 'on')
    use_gpu = os.getenv('USE_GPU', 'True').lower() in ('true', '1', 'yes', 'on')

    process_movies(
        base_path=movie_library_path,
        clip_length=clip_length,
        start_buffer=start_buffer,
        end_ignore_pct=end_ignore_pct,
        force=force,
        use_gpu=use_gpu
    )
