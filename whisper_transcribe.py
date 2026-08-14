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

from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from transformers import logging as transformers_logging

class WhisperRealTime:
    def __init__(self, model_id="openai/whisper-large-v3-turbo", cache_dir="./models"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Reduce noisy transformer warnings for generation behavior
        transformers_logging.set_verbosity_error()
        self.processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, 
            cache_dir=cache_dir
        ).to(self.device)
        # Clear generation config entries that create internal logits processors
        # to avoid duplicate-processor warnings from `generate()`.
        try:
            gen_cfg = self.model.generation_config
            if hasattr(gen_cfg, "forced_decoder_ids"):
                gen_cfg.forced_decoder_ids = None
            if hasattr(gen_cfg, "suppress_tokens"):
                gen_cfg.suppress_tokens = None
            if hasattr(gen_cfg, "begin_suppress_tokens"):
                gen_cfg.begin_suppress_tokens = None
            if hasattr(gen_cfg, "suppress_tokens_at_beginning"):
                gen_cfg.suppress_tokens_at_beginning = None
        except Exception:
            pass
        
    def transcribe(self, audio_array, sampling_rate=16000, translate=True, language="en"):
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

        # Choose Whisper generation task: 'translate' -> translate to target language,
        # 'transcribe' -> transcribe in detected language. Default: translate to English.
        task = "translate" if translate else "transcribe"

        with torch.no_grad():
            predicted_ids = self.model.generate(
                inputs.input_features,
                task=task,
                language=language
            )
            
        transcription = self.processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        return transcription.strip()