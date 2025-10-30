# HW2: Controllable Text-to-Music Generation

## Setup

```bash
uv sync
```

## Run

```bash
uv run python main_pipeline.py --config config_final.json
```

## Configuration

Edit `config_final.json` to specify:
- `target_dir`: path to target music files
- `reference_dir`: path to reference music files
- `output_dir`: where to save results
- `caption_model`: "qwen2-audio-natural" or "lp-musiccaps"
- `generators`: ["jasco"] or ["musicgen-melody"]
