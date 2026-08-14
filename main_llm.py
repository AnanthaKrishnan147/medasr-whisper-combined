import os
import queue
import threading
import time
import importlib
import torch
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
CHUNK_DURATION = 5  # Legacy variable, kept for reference
audio_queue = queue.Queue()
# Raw small frames from the audio callback; VAD worker will turn them into chunks
raw_frames = queue.Queue()

# Control primitives for starting/stopping the stream and processing
stream = None
processing_thread = None
vad_thread = None
running_event = threading.Event()
stream_lock = threading.Lock()

# Silero VAD utilities (loaded lazily)
silero_vad = None
silero_utils = None
silero_available = False

def audio_callback(indata, frames, time, status):
    """Receives audio chunks from the microphone in a background thread."""
    if status:
        print(f"Audio Status: {status}", flush=True)
    # Flatten the 2D audio array to 1D and put it in the processing queue
    # Only queue raw frames while running; VAD worker will form complete chunks
    if running_event.is_set():
        raw_frames.put(indata.copy().flatten())

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
def _process_loop():
    """Background loop that processes audio chunks from the queue."""
    while running_event.is_set():
        try:
            audio_array = audio_queue.get(timeout=0.5)
        except queue.Empty:
            continue

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


def _vad_worker(sampling_rate=SAMPLE_RATE,
                frame_duration_s=0.25,
                silence_timeout_s=1.0,
                min_speech_s=0.6):
    """Consume raw microphone frames and send a complete chunk only after speech has ended.
    This uses a small rolling window: while speech is detected in the latest window,
    we keep buffering. When silence persists for `silence_timeout_s`, we emit the
    accumulated speech chunk to `audio_queue`.
    """
    global silero_vad, silero_utils, silero_available

    frame_samples = int(frame_duration_s * sampling_rate)
    silence_samples = int(silence_timeout_s * sampling_rate)
    min_speech_samples = int(min_speech_s * sampling_rate)

    # Buffer only the current utterance while a speech event is active.
    speech_buffer = np.zeros(0, dtype=np.float32)
    speaking = False
    silence_counter = 0

    if not silero_available:
        try:
            local_silero = os.path.join(os.getcwd(), 'models', 'silero-vad')
            if os.path.isdir(local_silero):
                silero_vad, silero_utils = torch.hub.load(local_silero, 'silero_vad', source='local')
            else:
                silero_vad, silero_utils = torch.hub.load('snakers4/silero-vad', 'silero_vad')
            get_speech_timestamps = silero_utils[0]
            silero_utils = (get_speech_timestamps, silero_utils[2], silero_utils[3])
            silero_available = True
        except Exception as e:
            print(f"[VAD Init Warning]: {e}")
            silero_available = False

    while running_event.is_set():
        try:
            frame = raw_frames.get(timeout=0.2)
        except queue.Empty:
            time.sleep(0.05)
            continue

        speech_buffer = np.concatenate((speech_buffer, frame))

        # Keep a practical cap on the utterance buffer.
        max_utterance_len = sampling_rate * 30
        if speech_buffer.shape[0] > max_utterance_len:
            speech_buffer = speech_buffer[-max_utterance_len:]

        # Evaluate speech presence using a rolling window around the most recent data.
        window = speech_buffer[-int(1.0 * sampling_rate):]
        try:
            if silero_available:
                get_speech_timestamps = silero_utils[0]
                # FIX: Convert numpy array to torch tensor
                tensor_window = torch.from_numpy(window)
                timestamps = get_speech_timestamps(tensor_window, silero_vad, sampling_rate)
            else:
                timestamps = []
        except Exception as e:
            # Added error print to catch failures here
            print(f"[VAD Processing Warning]: {e}")
            timestamps = []

        # Energy fallback if VAD fails or is unavailable.
        if not timestamps and window.size >= int(0.03 * sampling_rate):
            tail = window[-int(0.03 * sampling_rate):]
            rms = np.sqrt(np.mean(tail ** 2))
            timestamps = [{'start': 0, 'end': len(window)}] if rms > 0.01 else []

        if timestamps:
            speaking = True
            silence_counter = 0
        else:
            if speaking:
                silence_counter += len(frame)
            else:
                # keep the buffer small while idle
                if speech_buffer.shape[0] > frame_samples * 4:
                    speech_buffer = speech_buffer[-frame_samples * 4:]

        if speaking and silence_counter >= silence_samples:
            if speech_buffer.shape[0] >= min_speech_samples:
                # Emit a complete utterance only once, and only after real silence.
                audio_queue.put(speech_buffer.copy())
            speech_buffer = np.zeros(0, dtype=np.float32)
            speaking = False
            silence_counter = 0


def start_transcription():
    """Start the microphone stream and processing thread."""
    global stream, processing_thread, vad_thread
    with stream_lock:
        if running_event.is_set():
            print("Transcription already running.")
            return

        # Create stream if needed
        if stream is None:
            # FIX: Change blocksize to small chunks matching the VAD window (0.2 seconds)
            stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32',
                                    blocksize=int(SAMPLE_RATE * 0.2),
                                    callback=audio_callback)

        # Start running and the processing thread
        running_event.set()
        stream.start()

        # Start processing and VAD workers
        processing_thread = threading.Thread(target=_process_loop, daemon=True)
        processing_thread.start()

        # start vad worker
        vad_thread = threading.Thread(target=_vad_worker, daemon=True)
        vad_thread.start()
        print("Microphone stream started.")


def stop_transcription():
    """Stop the microphone stream and processing thread; clear queued audio."""
    global stream, processing_thread, vad_thread
    with stream_lock:
        if not running_event.is_set():
            print("Transcription is not running.")
            return

        # Signal processing thread to stop
        running_event.clear()

        # Stop stream if available
        try:
            if stream is not None:
                stream.stop()
        except Exception as e:
            print(f"Error stopping stream: {e}")

        # Clear queued audio so no old chunks are processed on restart
        try:
            while not audio_queue.empty():
                audio_queue.get_nowait()
        except Exception:
            pass

        # also clear raw frames and stop vad thread
        try:
            while not raw_frames.empty():
                raw_frames.get_nowait()
        except Exception:
            pass

        # give vad thread a moment to exit
        try:
            if vad_thread is not None:
                vad_thread.join(timeout=1.0)
        except Exception:
            pass

        # Give processing thread a moment to exit
        if processing_thread is not None:
            processing_thread.join(timeout=1.0)

        print("Microphone stream stopped.")


if __name__ == '__main__':
    print("\nTranscription controller ready.")
    print("Type 'start' to begin, 'stop' to stop, 'quit' to exit.")
    try:
        while True:
            cmd = input('> ').strip().lower()
            if cmd == 'start':
                start_transcription()
            elif cmd == 'stop':
                stop_transcription()
            elif cmd in ('quit', 'exit'):
                stop_transcription()
                break
            elif cmd == 'status':
                print('running' if running_event.is_set() else 'stopped')
            else:
                print("Commands: start, stop, status, quit")
    except KeyboardInterrupt:
        stop_transcription()
        print('\nExited.')