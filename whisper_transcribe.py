import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

class WhisperRealTime:
    def __init__(self, model_id="openai/whisper-large-v3-turbo", cache_dir="./models"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, 
            cache_dir=cache_dir
        ).to(self.device)
        
    def transcribe(self, audio_array, sampling_rate=16000):
        # Convert raw audio numpy array to tensors
        inputs = self.processor(
            audio_array, 
            sampling_rate=sampling_rate, 
            return_tensors="pt"
        ).to(self.device)
        # Ensure the input tensor dtype matches the model parameters (fp16 vs fp32)
        model_dtype = next(self.model.parameters()).dtype
        if inputs.input_features.dtype != model_dtype:
            inputs.input_features = inputs.input_features.to(dtype=model_dtype, device=self.device)

        with torch.no_grad():
            predicted_ids = self.model.generate(inputs.input_features)
            
        transcription = self.processor.batch_decode(
            predicted_ids, 
            skip_special_tokens=True
        )[0]
        return transcription.strip()