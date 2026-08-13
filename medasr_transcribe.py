import torch
from transformers import AutoProcessor, AutoModelForCTC

class MedASRRealTime:
    def __init__(self, model_id="google/medasr", cache_dir="./models"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)
        self.model = AutoModelForCTC.from_pretrained(
            model_id, 
            cache_dir=cache_dir
        ).to(self.device)
        
    def transcribe(self, audio_array, sampling_rate=16000):
        # CTC models process inputs slightly differently than Seq2Seq models
        inputs = self.processor(
            audio_array, 
            sampling_rate=sampling_rate, 
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            logits = self.model(inputs.input_features).logits
            
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0]
        return transcription.strip()