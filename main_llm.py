import os
import queue
import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
from openai import OpenAI

# Import the transcription classes from your local files
from whisper_transcribe import WhisperRealTime
from medasr_transcribe import MedASRRealTime

# ==========================================
# 1. Setup Environment and OpenAI Client
# ==========================================
load_dotenv()

# Initialize the OpenAI client (automatically retrieves OPENAI_API_KEY from environment)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ==========================================
# 2. Load ASR Models
# ==========================================
print("Loading models into memory... This may take a moment.")
whisper_asr = WhisperRealTime()
medasr_asr = MedASRRealTime()
print("Models loaded successfully.")

# ==========================================
# 3. Audio Streaming Architecture
# ==========================================
SAMPLE_RATE = 16000
CHUNK_DURATION = 5  # Capture 5 seconds of audio at a time
audio_queue = queue.Queue()

def audio_callback(indata, frames, time, status):
    """Receives audio chunks from the microphone in a background thread."""
    if status:
        print(f"Audio Status: {status}", flush=True)
    # Flatten the 2D audio array to 1D and put it in the processing queue
    audio_queue.put(indata.copy().flatten())

def synthesize_transcripts(whisper_text, medasr_text):
    """Passes both transcripts to OpenAI to get a final, unified output."""
    if not whisper_text and not medasr_text:
        return "No speech detected."
        
    prompt = f"""
    You are an expert medical transcriptionist. I have two speech-to-text outputs 
    from the same audio clip. 
    
    Whisper (General): "{whisper_text}"
    MedASR (Medical): "{medasr_text}"
    
    Combine them into a single, highly accurate transcript. Prioritize MedASR for 
    complex medical terminology and Whisper for general conversational context, 
    grammar, and punctuation. 
    
    Output ONLY the final corrected transcript, nothing else.
    """
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # Fast and cost-effective; swap to "gpt-4o" for complex synthesis
        messages=[
            {"role": "system", "content": "You are a precise medical transcription editor."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )
    
    return response.choices[0].message.content.strip()

# ==========================================
# 4. Main Event Loop
# ==========================================
print("\nStarting real-time microphone stream. Press Ctrl+C to stop.")
print("Listening...")

try:
    # Open the microphone stream
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32', 
                        blocksize=int(SAMPLE_RATE * CHUNK_DURATION), 
                        callback=audio_callback):
        while True:
            # Wait for and retrieve the next 5-second audio chunk from the queue
            audio_array = audio_queue.get()
            print("\n--- Processing New Audio Chunk ---")
            
            # Transcribe via both models
            whisper_text = whisper_asr.transcribe(audio_array)
            print(f"[Whisper]: {whisper_text}")
            
            medasr_text = medasr_asr.transcribe(audio_array)
            print(f"[MedASR] : {medasr_text}")
            
            # Synthesize final output
            if whisper_text or medasr_text:
                final_output = synthesize_transcripts(whisper_text, medasr_text)
                print(f"\n✅ [FINAL LLM OUTPUT]: {final_output}\n")
                
except KeyboardInterrupt:
    print("\nTranscription stopped by user.")