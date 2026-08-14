# Combined Whisper + MedASR Transcription and LLM Runner

A small collection of scripts to run automatic speech recognition (ASR) using OpenAI Whisper and Google MedASR models, and to run a simple LLM entrypoint. This repository contains utility scripts, cached model snapshots, and an included Python virtual environment for reproducible runs.

## Contents
- `load_models.py` — helper to load model artifacts
- `main_llm.py` — entrypoint for running the LLM-related workflow
- `whisper_transcribe.py` — script to transcribe audio using Whisper
- `medasr_transcribe.py` — script to transcribe audio using MedASR
- `models/` — cached model data (snapshots/blobs)
- `asr_venv/` — included Python 3.10 virtual environment (optional)

## Requirements
- Linux (developed and tested on Linux)
- Python 3.10+
- Typical dependencies include `torch`, `transformers`, and `openai` or other ASR tooling. If you maintain a `requirements.txt` in this repo, install it; otherwise install the packages used by the scripts.

## Setup
1. Option A — Use the included virtualenv

```bash
source asr_venv/bin/activate
```

2. Option B — Create and activate a new virtualenv

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt  # if present
```

3. If you do not have `requirements.txt`, install the packages your environment requires. Example:

```bash
pip install torch transformers openai soundfile
```

## Usage

Transcribe audio with Whisper (example):

```bash
python whisper_transcribe.py --input path/to/audio.wav
```

Transcribe audio with MedASR (example):

```bash
python medasr_transcribe.py --input path/to/audio.wav
```

Run the LLM runner (example):

```bash
python main_llm.py --prompt "Summarize the following transcript"
```

If a script supports `--help`, run it to see available options:

```bash
python whisper_transcribe.py --help
```

## Models

Model artifacts are stored under `models/`. This repo includes cached snapshots for some models; the exact layout and filenames depend on how you downloaded or cached the models (e.g., via the Hugging Face hub).

## Notes
- The included `asr_venv/` is provided to simplify reproducible runs. It may already include many required packages.
- Adjust GPU/CPU-related options in the scripts if needed for your environment.

## Next steps / Suggestions
- Add a `requirements.txt` capturing the environment used to run these scripts.
- Add example audio files and a small end-to-end demo script.
- Add tests or a CI workflow to validate basic transcription functionality.

## License
Check project or dataset licenses for included model weights. No license is added to this repository by default.

---

If you want, I can also: add a `requirements.txt`, update individual script `--help` output, or commit these changes.
