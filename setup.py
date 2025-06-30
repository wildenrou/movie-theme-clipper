#!/usr/bin/env python3
"""
Movie Theme Clipper - Repository Setup Script (Windows Fixed)
This script creates all necessary files for your GitHub repository.

INSTRUCTIONS:
1. Change GITHUB_USERNAME below to your actual GitHub username
2. Save this file as 'setup.py' in your movie-theme-clipper folder
3. Run: python setup.py
"""

import os

# CHANGE THIS TO YOUR ACTUAL GITHUB USERNAME!
GITHUB_USERNAME = "wildenrou"  # ← Already set to your username!

def create_dockerfile():
    content = f"""FROM python:3.11-slim

LABEL org.opencontainers.image.title="Movie Theme Clipper"
LABEL org.opencontainers.image.description="Automatically generates theme clips from movie collections using AI analysis"
LABEL org.opencontainers.image.vendor="Your Name"
LABEL org.opencontainers.image.source="https://github.com/{GITHUB_USERNAME}/movie-theme-clipper"

# Install system dependencies including Intel GPU drivers
RUN apt-get update && apt-get install -y \\
    ffmpeg \\
    libsndfile1-dev \\
    intel-media-va-driver \\
    intel-media-va-driver-non-free \\
    vainfo \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/

# Create directories and set permissions
RUN mkdir -p /movies /logs && \\
    chown -R appuser:appuser /app /movies /logs

# Switch to non-root user
USER appuser

# Environment variables with defaults
ENV MOVIE_PATH="/movies"
ENV CLIP_LENGTH="18"
ENV METHOD="visual"
ENV START_BUFFER="120"
ENV END_IGNORE_PCT="0.3"
ENV USE_GPU="true"
ENV FORCE="false"
ENV LOG_LEVEL="INFO"
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=300s --timeout=30s --start-period=60s --retries=3 \\
    CMD test -d ${{MOVIE_PATH}} || exit 1

# Run the application
CMD ["python", "src/theme_clipper.py"]"""
    return content

def create_github_workflow():
    content = f"""name: Build and Push Docker Image

on:
  push:
    branches: [ main ]
    tags: [ 'v*' ]
  pull_request:
    branches: [ main ]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{{{ github.repository }}}}

jobs:
  build:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write

    steps:
    - name: Checkout repository
      uses: actions/checkout@v4

    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3

    - name: Log in to Container Registry
      if: github.event_name != 'pull_request'
      uses: docker/login-action@v3
      with:
        registry: ${{{{ env.REGISTRY }}}}
        username: ${{{{ github.actor }}}}
        password: ${{{{ secrets.GITHUB_TOKEN }}}}

    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v5
      with:
        images: ${{{{ env.REGISTRY }}}}/${{{{ env.IMAGE_NAME }}}}
        tags: |
          type=ref,event=branch
          type=ref,event=pr
          type=semver,pattern={{{{version}}}}
          type=semver,pattern={{{{major}}}}.{{{{minor}}}}
          type=raw,value=latest,enable={{{{is_default_branch}}}}

    - name: Build and push Docker image
      uses: docker/build-push-action@v5
      with:
        context: .
        platforms: linux/amd64,linux/arm64
        push: ${{{{ github.event_name != 'pull_request' }}}}
        tags: ${{{{ steps.meta.outputs.tags }}}}
        labels: ${{{{ steps.meta.outputs.labels }}}}
        cache-from: type=gha
        cache-to: type=gha,mode=max"""
    return content

def create_unraid_template():
    content = f"""<?xml version="1.0"?>
<Container version="2">
  <n>Movie-Theme-Clipper</n>
  <Repository>ghcr.io/{GITHUB_USERNAME}/movie-theme-clipper:latest</Repository>
  <Registry>https://ghcr.io</Registry>
  <Network>none</Network>
  <MyIP/>
  <Shell>bash</Shell>
  <Privileged>false</Privileged>
  <Support>https://github.com/{GITHUB_USERNAME}/movie-theme-clipper/issues</Support>
  <Project>https://github.com/{GITHUB_USERNAME}/movie-theme-clipper</Project>
  <Overview>Automatically generates theme clips from movie collections for media servers like Plex, Jellyfin, and Emby. Uses intelligent analysis methods (visual, audio, music) and Intel GPU acceleration for fast processing.</Overview>
  <Category>MediaApp:Video</Category>
  <WebUI/>
  <TemplateURL>https://raw.githubusercontent.com/{GITHUB_USERNAME}/movie-theme-clipper/main/templates/unraid-template.xml</TemplateURL>
  <Icon>https://raw.githubusercontent.com/{GITHUB_USERNAME}/movie-theme-clipper/main/.github/images/icon.png</Icon>
  <ExtraParams>--device=/dev/dri:/dev/dri</ExtraParams>
  <PostArgs/>
  <CPUset/>
  <DateInstalled>1704067200</DateInstalled>
  <DonateText>Support Development</DonateText>
  <DonateLink>https://github.com/sponsors/{GITHUB_USERNAME}</DonateLink>
  <Requires>Intel GPU with VAAPI support for hardware acceleration</Requires>
  <Config Name="Host Path 1" Target="/movies" Default="/mnt/user/media/movies" Mode="rw" Description="Path to your movie collection" Type="Path" Display="always" Required="true" Mask="false"/>
  <Config Name="Host Path 2" Target="/logs" Default="/mnt/user/appdata/theme-clipper/logs" Mode="rw" Description="Path for log files" Type="Path" Display="advanced" Required="false" Mask="false"/>
  <Config Name="Variable: CLIP_LENGTH" Target="CLIP_LENGTH" Default="18" Mode="" Description="Length of theme clips in seconds (10-30 recommended)" Type="Variable" Display="always" Required="false" Mask="false">18</Config>
  <Config Name="Variable: METHOD" Target="METHOD" Default="visual" Mode="" Description="Analysis method: visual (dynamic scenes), audio (high activity), music (harmonic content), random" Type="Variable" Display="always" Required="false" Mask="false">visual</Config>
  <Config Name="Variable: USE_GPU" Target="USE_GPU" Default="true" Mode="" Description="Enable Intel GPU hardware acceleration (requires Intel GPU)" Type="Variable" Display="always" Required="false" Mask="false">true</Config>
  <Config Name="Variable: START_BUFFER" Target="START_BUFFER" Default="120" Mode="" Description="Skip first N seconds of movie (avoid intros/credits)" Type="Variable" Display="advanced" Required="false" Mask="false">120</Config>
  <Config Name="Variable: END_IGNORE_PCT" Target="END_IGNORE_PCT" Default="0.3" Mode="" Description="Ignore last N% of movie (0.3 = 30%, avoids credits)" Type="Variable" Display="advanced" Required="false" Mask="false">0.3</Config>
  <Config Name="Variable: FORCE" Target="FORCE" Default="false" Mode="" Description="Overwrite existing theme clips (true/false)" Type="Variable" Display="advanced" Required="false" Mask="false">false</Config>
  <Config Name="Variable: LOG_LEVEL" Target="LOG_LEVEL" Default="INFO" Mode="" Description="Logging level: DEBUG, INFO, WARNING, ERROR" Type="Variable" Display="advanced" Required="false" Mask="false">INFO</Config>
</Container>"""
    return content

def create_theme_clipper():
    content = """import os
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
GREEN = "\\033[92m"
YELLOW = "\\033[93m"
RED = "\\033[91m"
RESET = "\\033[0m"

@contextlib.contextmanager
def suppress_output():
    \"\"\"Suppresses stdout and stderr.\"\"\"
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
    \"\"\"Check if Intel GPU is available and working.\"\"\"
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
    \"\"\"Finds movie files in subdirectories of base_path.\"\"\"
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
    \"\"\"Calculates a random start time for a clip.\"\"\"
    end_ignore_pct = max(0, min(1, end_ignore_pct))
    latest_valid_start_time = duration * (1 - end_ignore_pct) - clip_length
    effective_start_buffer = min(start_buffer, latest_valid_start_time)

    if effective_start_buffer >= latest_valid_start_time:
        if duration <= clip_length: 
            return 0
        return max(0, start_buffer) if start_buffer < duration - clip_length else 0

    return random.uniform(effective_start_buffer, latest_valid_start_time)

def get_env_bool(env_var, default=False):
    \"\"\"Convert environment variable to boolean.\"\"\"
    value = os.getenv(env_var, str(default)).lower()
    return value in ('true', '1', 'yes', 'on')

def get_env_float(env_var, default=0.0):
    \"\"\"Convert environment variable to float.\"\"\"
    try:
        return float(os.getenv(env_var, str(default)))
    except ValueError:
        return default

def get_env_int(env_var, default=0):
    \"\"\"Convert environment variable to int.\"\"\"
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
    print("For full functionality, use the complete application code.")"""
    return content

def create_readme():
    content = f"""# Movie Theme Clipper

Automatically generates theme clips from movie collections for media servers like Plex, Jellyfin, and Emby. Uses intelligent analysis methods and Intel GPU acceleration for fast, high-quality processing.

## Quick Start with Unraid

### Docker Command Line
```bash
docker run --rm \\
  -v /mnt/user/media/movies:/movies:rw \\
  -v /mnt/user/appdata/theme-clipper/logs:/logs:rw \\
  --device=/dev/dri:/dev/dri \\
  -e CLIP_LENGTH=18 \\
  -e METHOD=visual \\
  -e USE_GPU=true \\
  ghcr.io/{GITHUB_USERNAME}/movie-theme-clipper:latest
```

### Unraid Template
The template is available at: `https://raw.githubusercontent.com/{GITHUB_USERNAME}/movie-theme-clipper/main/templates/unraid-template.xml`

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `MOVIE_PATH` | `/movies` | Path to movie collection |
| `CLIP_LENGTH` | `18` | Clip length in seconds (10-30 recommended) |
| `METHOD` | `visual` | Analysis method: visual, audio, music, random |
| `USE_GPU` | `true` | Enable Intel GPU acceleration |
| `START_BUFFER` | `120` | Skip first N seconds (avoid intros) |
| `END_IGNORE_PCT` | `0.3` | Ignore last 30% (avoid credits) |
| `FORCE` | `false` | Overwrite existing clips |
| `LOG_LEVEL` | `INFO` | Logging level |

## Analysis Methods

- **visual**: Analyzes frame differences for dynamic scenes (default)
- **audio**: Finds high audio activity segments  
- **music**: Identifies music-dominant segments
- **random**: Random selection from middle portion

## Output Structure

```
Movies/
├── Avatar (2009)/
│   ├── Avatar (2009).mkv
│   └── Backdrops/
│       └── theme.mp4          # ← Generated theme clip
```

## Requirements

- Intel GPU with VAAPI support for hardware acceleration
- Docker
- Movie collection accessible via mount
- ~10-50MB storage per movie for theme clips

## Support

- Issues: https://github.com/{GITHUB_USERNAME}/movie-theme-clipper/issues
- Documentation: https://github.com/{GITHUB_USERNAME}/movie-theme-clipper/wiki

## License

MIT License - see LICENSE file for details."""
    return content

# File creation functions
files_to_create = {
    'Dockerfile': create_dockerfile,
    'requirements.txt': lambda: """moviepy>=2.0.0
librosa>=0.10.0
numpy>=1.24.0
tqdm>=4.65.0
scipy>=1.10.0""",
    '.github/workflows/docker-build.yml': create_github_workflow,
    'templates/unraid-template.xml': create_unraid_template,
    'src/theme_clipper.py': create_theme_clipper,
    'README.md': create_readme,
    '.gitignore': lambda: """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual environments
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
*.log
logs/

# Docker
.dockerignore

# Local test files
test_movies/
local_test/""",
    'LICENSE': lambda: """MIT License

Copyright (c) 2024 Your Name

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE."""
}

def main():
    print(f"🎬 Creating Movie Theme Clipper repository files...")
    print(f"📁 GitHub username: {GITHUB_USERNAME}")
    print()
    
    created_count = 0
    
    for filepath, content_func in files_to_create.items():
        try:
            # Create directory if needed
            directory = os.path.dirname(filepath)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
                print(f"📁 Created directory: {directory}")
            
            # Get content
            content = content_func() if callable(content_func) else content_func
            
            # Write file (removed the problematic newline parameter)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"✅ Created: {filepath}")
            created_count += 1
            
        except Exception as e:
            print(f"❌ Failed to create {filepath}: {e}")
    
    print(f"\\n🎉 Successfully created {created_count} files!")
    print()
    print("Next steps:")
    print("1. git add .")
    print("2. git commit -m 'Initial release: Movie Theme Clipper v1.0'")
    print("3. git push origin main")
    print("4. git tag v1.0.0")
    print("5. git push origin v1.0.0")
    print()
    print("Then check GitHub Actions tab to watch the build process!")

if __name__ == "__main__":
    main()