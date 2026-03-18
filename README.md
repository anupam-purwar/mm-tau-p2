# $\text{MM-Tau-p}^2$

MMTau is a multi-modal benchmarking framework for evaluating voice-based agentic systems on customer-service tasks. It extends [Tau$^2$-bench](https://github.com/sierra-research/tau2-bench) by introducing a full voice pipeline (ASR + LLM + TTS) that converts text-based agent interactions into realistic spoken conversations, then evaluates how well agents perform under the noise and ambiguity inherent in speech.

## How It Works

1. **Task retrieval** — Tasks (customer-service scenarios) are loaded from Tau$^2$-bench for a chosen domain (`retail` or `telecom`).
2. **Ground-truth generation** — A standard text-only Tau$^2$-bench orchestrator produces the ideal conversation trace.
3. **Voice simulation** — The same task is replayed through the voice pipeline: user utterances are synthesised to audio (TTS), the audio is transcribed (ASR), the agent reasons over the transcription (LLM + tool use), and the agent's reply is synthesised back to speech.
4. **Evaluation** — Both message-level and conversation-level LLM-as-judge evaluators score the voice conversation against the ground-truth, producing metrics such as pass@1, ARGA, clarification precision/recall, safety precision/recall, turn overhead, and more.

## Project Structure

```
MMTau/
├── run.py                       # CLI entry-point
├── config.py                    # All configurable parameters
├── requirements.txt             # Python dependencies
├── pipeline/
│   ├── pipe.py                  # VoicePipeline — orchestrates ASR → LLM → TTS
│   ├── ASR.py                   # Whisper and Parakeet speech recognition
│   ├── TTS.py                   # NeuTTS and ElevenLabs speech synthesis
│   ├── LLM.py                   # OpenAI chat-completion wrapper
│   └── utils.py                 # File I/O, WAV stitching, streaming helpers
├── simulator/
│   ├── Tau2BenchSimulator.py    # Tau2VoiceAgent & Tau2VoiceSimulator
│   └── MMTauEval.py             # Message, Conversation, and aggregate evaluators
└── Data/
    ├── audio/                   # Warmup audio, voice-cloning tensors & transcripts
    └── Prompts/                 # LLM-as-judge prompt templates & metric definitions
```

### Key Modules

| Module | Role |
|---|---|
| `run.py` | Parses CLI arguments, runs the simulation loop over a range of tasks, then invokes evaluation. |
| `config.py` | Central configuration: model names, devices, TTS/ASR backend selection, evaluation model, domain, persona injection flags, etc. |
| `pipeline/pipe.py` | `VoicePipeline` — sets up ASR, TTS, and LLM components; runs the main voice-conversation loop (record/read audio → transcribe → reason → synthesise). |
| `pipeline/ASR.py` | `Whisper` (HuggingFace) and `Parakeet` (NVIDIA NeMo) ASR backends with optional inverse text normalisation. |
| `pipeline/TTS.py` | `Neu` (NeuTTSAir, supports voice cloning) and `ElevenLabs` TTS backends with optional text normalisation and resampling. |
| `pipeline/LLM.py` | Thin OpenAI wrapper for chat completions (streaming and non-streaming). |
| `simulator/Tau2BenchSimulator.py` | `Tau2VoiceAgent` — extends both `VoicePipeline` and Tau$^2$-bench's `LLMAgent` to handle tool calls and persona injection inside a voice loop. `Tau2VoiceSimulator` — extends Tau$^2$-bench's `Orchestrator` to run paired ground-truth and voice conversations. |
| `simulator/MMTauEval.py` | `MessageAnalyzer` — per-message metric scoring. `ConversationAnalyzer` — conversation-level metrics (pass@k, ARGA, turn overhead). `MultiModalTauEval` — aggregate evaluator that produces a full report. |

## Usage

### Prerequisites

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Setup $\tau^2$ Base Agent [tau2-bench](https://github.com/sierra-research/tau2-bench)

   ```bash
   git clone https://github.com/sierra-research/tau2-bench.git
   cp -r tau2-bench/src/tau2 tau2
   cp -r tau2-bench/data/tau2 Data/tau2
   cd tau2-bench
   pip install -e .
   cd ..
   rm -rf tau2-bench
   export TAU2_DATA_DIR="Data"
   ```

   Add `TAU2_DATA_DIR="Data"` to your environment for future use

3. Set your OpenAI API key:

   ```bash
   export OPENAI_API_KEY="sk-..."
   ```

4. If using ElevenLabs TTS, set:

   ```bash
   export ELEVEN_API_KEY="..."
   ```

5. (Optional) 
   - For GPU-accelerated ASR/TTS, ensure CUDA is available and update `config.py` accordingly.
   - If using NeuTTS, install `espeak` **(required dependency)**
      Please refer to the following link for instructions on how to install `espeak`:
      https://github.com/espeak-ng/espeak-ng/blob/master/docs/guide.md

      ```bash
      # Mac OS
      brew install espeak

      # Ubuntu/Debian
      sudo apt install espeak
      ```
      
      Mac users may need to put the following lines at the top of the neutts.py file.

      ```python
      from phonemizer.backend.espeak.wrapper import EspeakWrapper
      _ESPEAK_LIBRARY = '/opt/homebrew/Cellar/espeak/1.48.04_1/lib/libespeak.1.1.48.dylib'  #use the Path to the library.
      EspeakWrapper.set_library(_ESPEAK_LIBRARY)
      ```

      Windows users may need to run (see https://github.com/bootphon/phonemizer/issues/163)

      ```pwsh
      $env:PHONEMIZER_ESPEAK_LIBRARY = "c:\Program Files\eSpeak NG\libespeak-ng.dll"
      $env:PHONEMIZER_ESPEAK_PATH = "c:\Program Files\eSpeak NG"
      setx PHONEMIZER_ESPEAK_LIBRARY "c:\Program Files\eSpeak NG\libespeak-ng.dll"
      setx PHONEMIZER_ESPEAK_PATH "c:\Program Files\eSpeak NG"
      ```

   - If using NeuTTS with GGUF based backbones ensure that `llama-cpp-python` is installed appropriately
      to build with CUDA support
      ```bash
         CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --no-cache-dir --force-reinstall
      ```
      without CUDA support
      ```bash
         pip install llama-cpp-python
      ```
   - If using NeuTTS with onnx decoder, ensure that `onnxruntime` is installed
      ```bash
         pip install onnxruntime
      ```
      If using CUDA GPU
      ```bash
         pip install onnxruntime-gpu
      ```
   - If using NeuTTS, install espeak


6. **Voice Resiliency**

   Edit the `Data/tau2/domains/telecom/main_policy.md` file and edit line 95 to specify format of phone numbers, do the same with other domains if needed

   `- Phone number (in the xxx-xxx-xxxx format make sure to include hyphens while making any query)`


### Running

```bash
python run.py <domain> [options]
```

`<domain>` is a required positional argument — either `retail` or `telecom`.

#### Options

| Flag | Default | Description |
|---|---|---|
| `--run_name NAME` | `MMTauPipeline` | Name prefix for the run directory under `.temp/`. |
| `--eval_only` | `False` | Skip simulation; only evaluate an existing run. |
| `--start N` | `0` | First task index to process. |
| `--stop N` | all tasks | Stop before this task index (exclusive). |
| `--split NAME` | `test` | Tau$^2$-bench task split to use. |
| `--inject-persona` | from config | Inject the user's persona (from task metadata) into the agent prompt after identity verification. |
| `--inject-context` | from config | Dynamically infer and inject user context from the ongoing conversation. |
| `--metrics-to-skip M [M ...]` | none | Metric names to exclude from evaluation. |
| `--critical-fields F [F ...]` | domain default | Override domain-specific critical fields for evaluation. |
| `--log-level LEVEL` | `DEBUG` | Python logging level. |
| `--duplicate` | `True` | Mirror log output to the terminal in addition to the log file. |
| `--save` | `True` | Retain intermediate files (audio, transcripts) after the run. |

#### Examples

Run all telecom tasks with persona injection:

```bash
python run.py telecom --inject-persona
```

Run a subset of retail tasks:

```bash
python run.py retail --start 0 --stop 10 --run_name retail_first10
```

Evaluate a previous run without re-simulating:

```bash
python run.py telecom --eval_only --run_name my_previous_run
```

### Output

Each run creates a directory at `.temp/<run_name>/` containing:

- `ground.txt` — ground-truth text conversation
- `implementation_llm.txt` — what the LLM intended to say/heard
- `implementation_actual.txt` — what was actually spoken/heard (post ASR/TTS)
- `*_user.wav` / `*_agent.wav` — individual turn audio files
- `pipeline.log` — full run log
- `Evals*.pkl` — pickled evaluation report