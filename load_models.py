import os
from dotenv import load_dotenv
from huggingface_hub import login
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

# ==========================================
# 4. Load Whisper Large v3 Turbo
# ==========================================
print("Loading Whisper Large v3 Turbo...")
whisper_id = "openai/whisper-large-v3-turbo"

# The cache_dir argument ensures it saves to and loads from ./models
whisper_processor = AutoProcessor.from_pretrained(
    whisper_id, 
    cache_dir=model_cache_dir
)
whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
    whisper_id, 
    cache_dir=model_cache_dir
)
print("Whisper loaded successfully.")

# ==========================================
# 5. Load Google MedASR
# ==========================================
print("\nLoading Google MedASR...")
medasr_id = "google/medasr"

medasr_processor = AutoProcessor.from_pretrained(
    medasr_id, 
    cache_dir=model_cache_dir
)
medasr_model = AutoModelForCTC.from_pretrained(
    medasr_id, 
    cache_dir=model_cache_dir
)
print("MedASR loaded successfully.")