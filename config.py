#ASR
ASR_WARMUP_FILE = 'Data/audio/WarmASR.wav'
WHISPER_MODEL = 'large-v3-turbo'
PARAKEET_MODEL = "nvidia/parakeet-tdt-0.6b-v2"
ASR_DEVICE = 'cpu' # ['cpu', 'cuda']

#TTS
NEUTTS_BACKBONE_MODEL = "neuphonic/neutts-air-q8-gguf"
NEUTTS_BACKBONE_DEVICE = 'cpu' # ['cpu', 'cuda'] # gguf models can only be used with cpu
NEUTTS_CODEC_MODEL = "neuphonic/neucodec-onnx-decoder"
NEUTTS_CODEC_DEVICE = 'cpu' # ['cpu', 'cuda']
CLONING_AUDIO_TENSOR_PATH = 'Data/audio/jo.pt'
CLONING_AUDIO_TRANSCRIPT_PATH = 'Data/audio/jo.txt'
ELEVENLABS_VOICE = '21m00Tcm4TlvDq8ikWAM'
ELEVENLABS_MODEL = "eleven_turbo_v2_5"
DESIRED_SAMPLE_RATE = 24000  # Target sample rate (Hz) for all generated audio

#PIPELINE
TEMP_DIR = '.temp' # Root directory for intermediate/output artifacts
ASR_TYPE = 'parakeet' # ['parakeet', 'whisper']
TTS_TYPE = 'neutts' # ['neutts', 'elevenlabs']
STREAM_AUDIO_GENERATION = False        # When True, TTS generates audio in streaming chunks

#EVALUATOR
EVAL_MODEL = 'gpt-4.1'
DOMAIN = 'telecom' # ['telecom', 'retail']
_CRITICAL_FIELDS = {
    'retail': ['OrderID', 'Username'],
    'telecom': ['Phone Number']
}
CRITICAL_FIELDS = _CRITICAL_FIELDS.get(DOMAIN, ValueError(f"Invalid domain {DOMAIN}"))

#GENERATOR
AGENT_MODEL = 'gpt-4.1'
AGENT_TEMPERATURE = 0.0
USER_MODEL = 'gpt-4.1'
USER_TEMPERATURE = 0.0
INJECT_PERSONA = True # Inject task-level user persona into the agent prompt
INJECT_CONTEXT = True # Inject inferred conversational context into the agent prompt
USER_CLONING_AUDIO_TENSOR_PATH = 'Data/audio/dave.pt'
USER_CLONING_AUDIO_TRANSCRIPT_PATH = 'Data/audio/dave.txt'
AGENT_CLONING_AUDIO_TENSOR_PATH = 'Data/audio/jo.pt'
AGENT_CLONING_AUDIO_TRANSCRIPT_PATH = 'Data/audio/jo.txt'
SEED = 67 # Random seed for reproducible Tau2 task initialization
MAX_STEPS = 100 # Maximum conversation turns before forced termination