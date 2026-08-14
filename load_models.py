import os
import subprocess
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import login
import torch

# Some installations of `torchaudio` (GPU wheels) can fail to load at import-time
# if the system doesn't have the matching CUDA runtime (libcudart). That causes
# a hard OSError when `transformers` tries `import torchaudio`. To make the
# loader robust, attempt to import `torchaudio` early and, on failure, insert a
# lightweight dummy module so `transformers` can import and we can still use the
# HF processors/models. If the project needs torchaudio functionality later,
# the user should install a compatible `torchaudio` wheel or the CPU-only build.
try:
    import torchaudio  # type: ignore
except Exception as _torchaudio_err:
    import types, sys, importlib.util
    dummy = types.ModuleType('torchaudio')
    def _raise_on_use(*args, **kwargs):
        raise RuntimeError(
            "torchaudio native extension failed to load. "
            "Install a compatible torchaudio (or CPU-only) wheel, or ensure CUDA libs are available. "
            f"Original error: {_torchaudio_err}")
    # provide minimal attributes used by downstream code to avoid import crashes
    dummy.load = _raise_on_use
    dummy.info = lambda *a, **k: None
    dummy.__version__ = '0.0.0'
    # Provide a minimal ModuleSpec so importlib.util.find_spec doesn't raise
    dummy.__spec__ = importlib.util.spec_from_loader('torchaudio', loader=None)
    dummy.__file__ = '<torchaudio_dummy>'
    dummy.__path__ = []
    sys.modules['torchaudio'] = dummy
    print("Warning: torchaudio extension failed to load; using a dummy fallback. If you need torchaudio, install a compatible wheel.")

from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, AutoModelForCTC

# 1. Load environment variables from the .env file
load_dotenv()

# 2. Authenticate with Hugging Face
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)
else:
    print("Warning: HF_TOKEN not found in .env file. Some models may require authentication.")

# 3. Define the local directory to store model weights
model_cache_dir = "./models"
os.makedirs(model_cache_dir, exist_ok=True)

def _cached_repo_path(model_id: str, cache_dir: str) -> str:
    # HF stores repos as models--owner--repo under the cache dir when using cache_dir
    repo_name = f"models--{model_id.replace('/', '--')}"
    return os.path.join(cache_dir, repo_name)


def _resolve_snapshot_path(repo_dir: str) -> str:
    """If the HF cache repo dir contains a `snapshots` directory, return the
    newest snapshot path; otherwise return the repo_dir itself.
    """
    snapshots_dir = os.path.join(repo_dir, 'snapshots')
    if os.path.isdir(snapshots_dir):
        # List snapshot subdirs and pick the most recent by name (or first)
        entries = sorted([os.path.join(snapshots_dir, d) for d in os.listdir(snapshots_dir) if os.path.isdir(os.path.join(snapshots_dir, d))])
        if entries:
            return entries[-1]
    return repo_dir

# ==========================================
# 4. Load Whisper Large v3 Turbo
# ==========================================
print("Loading Whisper Large v3 Turbo...")
whisper_id = "openai/whisper-large-v3-turbo"

# If the HF repo already exists in our local `models` folder, load from there
whisper_repo = _cached_repo_path(whisper_id, model_cache_dir)
if os.path.isdir(whisper_repo):
    snapshot_path = _resolve_snapshot_path(whisper_repo)
    print(f"Found cached Whisper in {whisper_repo}; loading from local cache snapshot {snapshot_path}.")
    whisper_processor = AutoProcessor.from_pretrained(snapshot_path)
    whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(snapshot_path)
else:
    # download into cache_dir
    whisper_processor = AutoProcessor.from_pretrained(
        whisper_id,
        cache_dir=model_cache_dir
    )
    whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        whisper_id,
        cache_dir=model_cache_dir
    )
    print(f"Whisper downloaded into cache at {model_cache_dir}.")

print("Whisper loaded successfully.")

# ==========================================
# 5. Load Google MedASR
# ==========================================
print("\nLoading Google MedASR...")
medasr_id = "google/medasr"

medasr_repo = _cached_repo_path(medasr_id, model_cache_dir)
if os.path.isdir(medasr_repo):
    snapshot_path = _resolve_snapshot_path(medasr_repo)
    print(f"Found cached MedASR in {medasr_repo}; loading from local cache snapshot {snapshot_path}.")
    medasr_processor = AutoProcessor.from_pretrained(snapshot_path)
    medasr_model = AutoModelForCTC.from_pretrained(snapshot_path)
else:
    medasr_processor = AutoProcessor.from_pretrained(
        medasr_id,
        cache_dir=model_cache_dir
    )
    medasr_model = AutoModelForCTC.from_pretrained(
        medasr_id,
        cache_dir=model_cache_dir
    )
    print(f"MedASR downloaded into cache at {model_cache_dir}.")

print("MedASR loaded successfully.")

# ==========================================
# 6. Ensure Silero VAD repo is present locally
# ==========================================
silero_dir = os.path.join(model_cache_dir, 'silero-vad')
if os.path.isdir(silero_dir):
    print(f"Found Silero VAD repo in {silero_dir}; using local copy.")
else:
    print(f"Cloning Silero VAD into {silero_dir}...")
    try:
        subprocess.run(['git', 'clone', '--depth', '1', 'https://github.com/snakers4/silero-vad', silero_dir], check=True)
        print("Silero VAD cloned successfully.")
    except Exception as e:
        print(f"Failed to clone Silero VAD: {e}")

# Load silero_vad from the local repo using torch.hub (source='local')
try:
    print("Loading Silero VAD from local repo (torch.hub)...")
    silero_vad = torch.hub.load(silero_dir, 'silero_vad', source='local')
    print("Silero VAD loaded.")
except Exception as e:
    print(f"Warning: failed to load Silero VAD locally: {e}")
    silero_vad = None

print("All models processed.")