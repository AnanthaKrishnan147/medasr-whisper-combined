import torch
# Guard against torchaudio native extension import failures (missing CUDA runtime).
try:
    import torchaudio  # type: ignore
except Exception as _torchaudio_err:
    import types, sys, importlib.util
    dummy = types.ModuleType('torchaudio')
    def _raise_on_use(*args, **kwargs):
        raise RuntimeError(
            "torchaudio native extension failed to load. Install a compatible torchaudio wheel or ensure CUDA libs are available. "
            f"Original error: {_torchaudio_err}")
    dummy.load = _raise_on_use
    dummy.info = lambda *a, **k: None
    dummy.__version__ = '0.0.0'
    dummy.__spec__ = importlib.util.spec_from_loader('torchaudio', loader=None)
    dummy.__file__ = '<torchaudio_dummy>'
    dummy.__path__ = []
    sys.modules['torchaudio'] = dummy

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