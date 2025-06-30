FROM python:3.11-slim

LABEL org.opencontainers.image.title="Movie Theme Clipper"
LABEL org.opencontainers.image.description="Automatically generates theme clips from movie collections using AI analysis"
LABEL org.opencontainers.image.vendor="wildenrou"
LABEL org.opencontainers.image.source="https://github.com/wildenrou/movie-theme-clipper"

# Install system dependencies including Intel GPU drivers
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1-dev \
    intel-media-va-driver \
    vainfo \
    curl \
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
RUN mkdir -p /movies /logs && \
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
HEALTHCHECK --interval=300s --timeout=30s --start-period=60s --retries=3 \
    CMD test -d ${MOVIE_PATH} || exit 1

# Run the application
CMD ["python", "src/theme_clipper.py"]