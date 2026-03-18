from neuttsair import NeuTTSAir
import soundfile as sf
import datetime
from torch import load as torch_load
from pipeline.utils import read_file
import librosa
import numpy as np

from typing import AsyncIterator, Optional
import asyncio
import base64
import json
import os
import websockets

from nemo_text_processing.text_normalization.normalize import Normalizer
import re

from config import *

import logging
logger = logging.getLogger(__name__)


class TTS:

    def __init__(self, use_text_norm = True):
        
        self.use_text_norm = use_text_norm
        self.text_norm = TextNormalizer()
        self._setup()
    
    # Backend specific implementation, must be overridden by subclasses
    def tts(self):
        raise NotImplementedError

    # Resample a waveform array from original SR to target SR
    def _resample(self, wav, original, target):
        wav =  librosa.resample(wav, orig_sr=original, target_sr=target)
        return wav
    
    def _warmup(self):
        file_out = self.tts('One Two Three Four Five Six Seven Eight Nine Ten Eleven Twelve Thirteen Fourteen Fifteen')
        os.remove(file_out)

    # Backend specific implementation, must be overridden by subclasses
    def _setup(self):
        pass

class Neu(TTS):

    default_sample_rate = None
    desired_sample_rate = DESIRED_SAMPLE_RATE

    def __init__(self, backbone_device = NEUTTS_BACKBONE_DEVICE, codec_device = NEUTTS_CODEC_DEVICE, sample_rate = 24000):

        super().__init__()
        self.backbone_device = backbone_device
        self.codec_device = codec_device
        self.desired_sample_rate = sample_rate

    def tts(self, input, context = None, stream = False, file_out = '', text_norm = True):
        """Synthesize speech and write to a WAV file.

        Args:
            input: Text to synthesize
            context: Voice-cloning context tuple (encoded_codes, transcript)
                     Falls back to the default context loaded during setup
            stream: Use streaming inference (generates audio chunk by chunk)
            file_out: Output WAV path; auto-generated if empty
            text_norm: Apply text normalization before synthesis

        Returns:
            Path to the generated WAV file
        """
        if text_norm:
            input = re.sub(r'\*', '', input)  # strip markdown-style asterisks
            input = self.text_norm(input)
        file_out = file_out if file_out else f"{TEMP_DIR}/{'.'+str(datetime.datetime.now()).replace(' ','-')}.wav"
        
        if not context:
            context = self.context
        if not stream:
            wav = self.model.infer(input, context[0], context[1])
            wav = self._resample(wav, original=self.default_sample_rate, target=int(self.desired_sample_rate))

        else:
            wav = np.array()
            for chunk in self.model.infer_stream(input, context[0], context[1]):
                chunk = self._resample(chunk, original=self.default_sample_rate, target=int(self.desired_sample_rate))
                wav = np.concatenate((wav, chunk), axis=0)
        
        sf.write(file_out, wav, self.desired_sample_rate)
        return file_out

    # Load the NeuTTS model and prepare the default voice-cloning context
    def _setup(self):
        self.model = NeuTTSAir(
                        backbone_repo = NEUTTS_BACKBONE_MODEL,
                        backbone_device = NEUTTS_BACKBONE_DEVICE,
                        codec_repo = NEUTTS_CODEC_MODEL,
                        codec_device = NEUTTS_CODEC_DEVICE,
                        )
        self.default_sample_rate = self.model.sample_rate

        self.context = self._get_context(ref_codes = CLONING_AUDIO_TENSOR_PATH, ref_text=read_file(CLONING_AUDIO_TRANSCRIPT_PATH))
        self.need_resampling = abs(self.desired_sample_rate - self.default_sample_rate) > 0
    
    def _get_context(self, audio_path = None, ref_codes = None, ref_text = None):
        if not ref_codes:
            ref_codes = self.model.encode_reference(audio_path)
        if isinstance(ref_codes, str):
            ref_codes = torch_load(ref_codes)
        return (ref_codes, ref_text)


class ElevenLabsBase(TTS):
    """Low-level async client for the ElevenLabs multi-stream WebSocket TTS API.

    Manages a persistent WebSocket connection with automatic reconnection.
    Each synthesis call creates a unique context so multiple requests can be
    multiplexed on the same socket.
    """
     
    DEFAULT_VOICE_ID = ELEVENLABS_VOICE

    def __init__(
        self,
        api_key: Optional[str] = None,
        voice_id: str = DEFAULT_VOICE_ID,
        model_id: str = ELEVENLABS_MODEL,
        sample_rate: int = 24000,
    ):
        """
        Args:
            api_key: ElevenLabs API key (falls back to ELEVENLABS_API_KEY env var)
            voice_id: ElevenLabs voice identifier
            model_id: TTS model to use on the ElevenLabs side
            sample_rate: Sample rate in Hz
        """
        self.api_key = api_key or os.getenv("ELEVENLABS_API_KEY")
        self.voice_id = voice_id
        self.model_id = model_id
        self.sample_rate = sample_rate
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._lock = asyncio.Lock() # Serializes concurrent synthesis calls
        self._context_counter = 0
        super().__init__()

    def _get_ws_url(self) -> str:
        params = [
            f"model_id={self.model_id}",
            f"output_format=pcm_{self.sample_rate}",
            "auto_mode=true",  # Enable auto mode for better short-text handling
        ]
        return f"wss://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}/multi-stream-input?{'&'.join(params)}"

    async def connect(self):
        if self._ws is not None:
            return
        headers = {"xi-api-key": self.api_key}
        self._ws = await websockets.connect(
            self._get_ws_url(),
            additional_headers=headers,
            ping_interval=20,
            ping_timeout=20,
            max_size=10 * 1024 * 1024,  # 10MB to handle large audio responses
        )
        init_msg = {
            "text": " ",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75,
                "speed": 1.0,
            },
            "generation_config": {
                "chunk_length_schedule": [50, 50, 75, 100],
            },
        }
        await self._ws.send(json.dumps(init_msg))

    async def close(self):
        if self._ws:
            try:
                await self._ws.send(json.dumps({"close_socket": True}))
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

    def _is_connected(self) -> bool:
        if self._ws is None:
            return False
        try:
            return self._ws.state.name == "OPEN"
        except Exception:
            return False

    async def _ensure_connected(self):
        if not self._is_connected():
            self._ws = None
            await self.connect()

    async def synthesize_stream(
        self,
        text: str,
    ) -> AsyncIterator[bytes]:
        """Stream PCM audio bytes for input text via the ElevenLabs WebSocket
        Yields raw PCM byte chunks as they arrive.  Each call uses a unique
        context_id so the caller can issue concurrent requests on the same socket
        """
        if not text or not text.strip():
            return

        async with self._lock:
            await self._ensure_connected()
            self._context_counter += 1
            context_id = f"ctx_{self._context_counter}"

            init_context = {
                "text": text + " ",  # trailing space signals end-of-input to the API
                "context_id": context_id,
                "flush": True,
            }
            await self._ws.send(json.dumps(init_context))

            close_context = {
                "context_id": context_id,
                "close_context": True,
            }
            await self._ws.send(json.dumps(close_context))

            while True:
                try:
                    msg = await asyncio.wait_for(self._ws.recv(), timeout=30.0)
                    data = json.loads(msg)

                    # Ignore messages from other contexts (multiplexed socket)
                    if data.get("contextId") != context_id:
                        continue

                    if data.get("isFinal"):
                        break

                    if "audio" in data:
                        audio_bytes = base64.b64decode(data["audio"])
                        if audio_bytes:
                            yield audio_bytes

                except asyncio.TimeoutError:
                    logger.warning("ElevenLabs WebSocket timeout")
                    break
                except Exception as e:
                    logger.error("ElevenLabs receive error: %s", e)
                    break

    # internally calls synthesize_stream and returns the concatenated byte stream
    async def synthesize(
        self,
        text: str,
        language: str = "en",
    ) -> bytes:
        if not text or not text.strip():
            return b""

        chunks = []
        async for chunk in self.synthesize_stream(text):
            chunks.append(chunk)
        return b"".join(chunks)


class ElevenLabs(ElevenLabsBase, TTS):
    """High-level TTS wrapper around ElevenLabsBase that writes WAV files
    Handles PCM-to-float conversion, optional resampling, and file output
    Text normalization is off by default because ElevenLabs normalizes server side
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        voice_id: str = ELEVENLABS_VOICE,
        model_id: str = ELEVENLABS_MODEL,
        sample_rate: int = 24000,
        desired_sample_rate: int = DESIRED_SAMPLE_RATE,
    ):
        self.desired_sample_rate = desired_sample_rate
        ElevenLabsBase.__init__(self, api_key, voice_id, model_id, sample_rate)
        TTS.__init__(self)

    async def tts(self, input, stream = True, file_out = None, text_norm = False):
        """Synthesize text to a wav file via the ElevenLabs API
            text_norm default value is set to False as ElevenLabs does text normalization at their end
        Args:
            input: Text to synthesize
            stream: Use streaming synthesis (processes audio chunk by chunk)
            file_out: Output WAV path; auto-generated if None
            text_norm: Apply local text normalization

        Returns:
            Path to the generated wav file
        """
        if text_norm:
            input = re.sub(r'\*', '', input)
            input = self.text_norm(input)

        file_out = file_out if file_out else f"{TEMP_DIR}/{'.'+str(datetime.datetime.now()).replace(' ','-')}.wav"

        if not stream:
            bytes_audio = await self.synthesize(input, language="en")
            wav = np.frombuffer(bytes_audio, dtype=np.int16) / 32767  # convert int16 PCM -> float32
            if self.need_resampling:
                wav = self._resample(wav, self.sample_rate, self.desired_sample_rate)

        else:
            wav = np.array()
            async for byte_chunk in self.synthesize_stream(input):
                audio_chunk = np.frombuffer(byte_chunk, dtype=np.int16) / 32767
                if self.need_resampling:
                    audio_chunk = self._resample(audio_chunk, self.sample_rate, self.desired_sample_rate)
                wav = np.concatenate((wav, audio_chunk), axis=0)

        sf.write(file_out, wav, self.desired_sample_rate)
        return file_out

    def _setup(self):
        self.need_resampling = abs(self.desired_sample_rate - self.sample_rate) > 0
        if self.need_resampling:
            logger.info("Resampling needed from %dHz to %dHz for playback", self.sample_rate, self.desired_sample_rate)


class TextNormalizer:

    def __init__(self):
        self.model = Normalizer(input_case='cased', lang='en')

    def __call__(self, text):
        return self.norm(text)
    
    def preprocess(self, text):
        text = text.lstrip()
        return text

    def norm(self, text:str):
        text = self.preprocess(text)
        text = self.model.normalize(text, verbose=False, punct_post_process=True)
        return text