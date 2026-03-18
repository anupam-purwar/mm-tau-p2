from torch.cuda import is_available as cuda_available
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import nemo.collections.asr as nemo_asr
from nemo_text_processing.inverse_text_normalization.inverse_normalize import InverseNormalizer

from config import *

import logging
logger = logging.getLogger(__name__)

class ASR:

    def __init__(self, inverse_norm=False):

        self.device = ASR_DEVICE
        self.inverse_norm = inverse_norm
        if inverse_norm:
            self.inverse_norm_model = InverseTextNorm()
    
    # Backend-specific implementation, must be overridden by subclasses
    def _asr(self, file):
        raise NotImplementedError
    
    def asr(self, file):
        output = self._asr(file)
        if self.inverse_norm:
            output = self.inverse_norm_model(output)
        return output
    
    # Warm-up before initial run to speed up generation
    def _warmup(self, test_file = ASR_WARMUP_FILE):
        try:
            self.asr(test_file)
        except Exception as e:
            logger.warning('Skipping ASR Warmup, demo file not found: %s', e)

class InverseTextNorm:

    def __init__(self, verbose=False):
        self.model = InverseNormalizer(lang='en')
        self.verbose = verbose

    def __call__(self, text):
        return self.inverse_norm(text)
    
    def preprocess(self, text):
        text = text.lstrip().rstrip()
        return text

    def inverse_norm(self, text:str):
        text = self.preprocess(text)
        text = self.model.inverse_normalize(text, verbose=self.verbose)
        return text


class Whisper(ASR):

    def __init__(self, model_name=WHISPER_MODEL, dtype='float32', inverse_norm=False):

        super().__init__(inverse_norm)
        self.model_name = model_name
        self.dtype = dtype
        self._setup()
    
    def _asr(self, file):

        with sdpa_kernel(SDPBackend.MATH):
            asr_result = self.model(file, generate_kwargs = {"language":"en"})['text'] # for running on gpu dont pass audio path, rather a copy of audio file, see torch.compile example https://huggingface.co/openai/whisper-large-v3-turbo
        return asr_result

    def _setup(self):

        torch.set_float32_matmul_precision("high")
        self.dtype = torch.float16 if cuda_available() else torch.float32
        model_id = f"openai/whisper-{self.model_name}"

        model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=self.dtype, low_cpu_mem_usage=True).to(self.device)
        model.generation_config.max_new_tokens = 256
        processor = AutoProcessor.from_pretrained(model_id)

        self.model = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            chunk_length_s=30,
            torch_dtype=self.dtype,
            device = self.device,
        )
        self._warmup()

class Parakeet(ASR):

    def __init__(self, model_name = PARAKEET_MODEL, inverse_norm=False):
        super().__init__(inverse_norm=inverse_norm)
        self.model_name = model_name
        self._setup()

    def _asr(self, file):

        if isinstance(file, str):
            output = self.model.transcribe([file])
            return output[0].text
        elif isinstance(file, list):
            output = self.model.transcribe(file)
            return [x.text for x in output]
        
    def _setup(self):
        self.model = nemo_asr.models.ASRModel.from_pretrained(model_name=self.model_name)
        self._warmup()