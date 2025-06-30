# Movie Theme Clipper

Automatically generates theme clips from movie collections for media servers like Plex, Jellyfin, and Emby. Uses intelligent analysis methods and Intel GPU acceleration for fast, high-quality processing.

## Quick Start with Unraid!

### Docker Command Line
```bash
docker run --rm \
  -v /mnt/user/media/movies:/movies:rw \
  -v /mnt/user/appdata/theme-clipper/logs:/logs:rw \
  --device=/dev/dri:/dev/dri \
  -e CLIP_LENGTH=18 \
  -e METHOD=visual \
  -e USE_GPU=true \
  ghcr.io/wildenrou/movie-theme-clipper:latest
```

### Unraid Template
The template is available at: `https://raw.githubusercontent.com/wildenrou/movie-theme-clipper/main/templates/unraid-template.xml`

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

- Issues: https://github.com/wildenrou/movie-theme-clipper/issues
- Documentation: https://github.com/wildenrou/movie-theme-clipper/wiki

## License

MIT License - see LICENSE file for details.