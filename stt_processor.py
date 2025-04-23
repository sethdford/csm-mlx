#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Whisper online & offline ASR based on faster-whisper transcription engine.

Source code modified from https://github.com/ufal/whisper_streaming

License MIT:
Copyright (c) 2024 Institute of Formal and Applied Linguistics, Faculty of Mathematics and Physics, Charles University
Copyright (c) 2024 Ondrej Platek

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import sys
import numpy as np
import logging
from typing import List, Tuple, Optional

# Set logging format before importing faster_whisper
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("stt_processor")

# Attempt to import faster_whisper AFTER setting logging config
try:
    from faster_whisper import WhisperModel
except ImportError:
    logger.error("faster_whisper not found. Please install it: pip install faster-whisper")
    sys.exit(1)

SAMPLING_RATE = 16000


def add_shared_args(parser):
    """ Add arguments shared by whisper_online.py and whisper_mic.py """
    parser.add_argument('--model-size', type=str, default="tiny.en",
                        help="Name of the Whisper model to use (e.g., tiny.en, base, small, medium, large, large-v2, large-v3).")
    parser.add_argument('--language', type=str, default="en",
                        help="Language code for transcription (e.g., en, de, es). Use 'auto' for language detection.")
    parser.add_argument('--device', type=str, default="cpu",
                        help="Device to use for computation (e.g., 'cpu', 'cuda', 'mps').")
    parser.add_argument('--compute-type', type=str, default="int8",
                        help="Compute type for the model (e.g., 'int8', 'float16', 'float32').")
    return parser

class FasterWhisperASR:
    """ Speech-to-text transcription using Faster Whisper."""

    # TODO: Consider adding options for beam_size, vad_filter, etc.
    def __init__(self, modelsize=None, lan=None, cache_dir=None, model_path=None, device="cpu", compute_type="int8"):
        self.language = lan
        self.model_path = model_path
        self.modelsize = modelsize
        self.cache_dir = cache_dir
        self.device = device
        self.compute_type = compute_type

        if model_path is not None:
            logger.info(f"Loading whisper model from path: {model_path}")
            self.model_size_or_path = model_path
        elif modelsize is not None:
            logger.info(f"Loading whisper model size: {modelsize}")
            self.model_size_or_path = modelsize
        else:
            raise ValueError("Either 'modelsize' or 'model_path' must be provided.")
        
        # Lazy load the model
        self.model = None

    def _load_model(self):
        """Loads the Whisper model if it hasn't been loaded yet."""
        if self.model is None:
            logger.info(f"Initializing Whisper model: {self.model_size_or_path}, Device: {self.device}, Compute Type: {self.compute_type}")
            try:
                self.model = WhisperModel(
                    self.model_size_or_path, 
                    device=self.device, 
                    compute_type=self.compute_type,
                    cache_dir=self.cache_dir,
                    # download_root=self.cache_dir # download_root is deprecated, cache_dir covers it
                )
                logger.info("Whisper model loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load Whisper model: {e}")
                raise # Re-raise the exception after logging

    def transcribe(self, audio, init_prompt=""):
        self._load_model() # Ensure model is loaded
        # logger.debug(f"Transcribing audio chunk of length {len(audio)/SAMPLING_RATE:.2f}s")
        # faster-whisper returns a generator, which we iterate over.
        # For non-VAD scenarios like this direct transcription, it usually yields one result.
        segments, info = self.model.transcribe(
            audio, 
            language=self.language if self.language != "auto" else None,
            initial_prompt=init_prompt,
            # VAD parameters are often set here for filtering, but we handle VAD outside
            # word_timestamps=False # Typically False for streaming, True if needed
        )
        # Concatenate text from all segments (usually just one)
        transcript = "".join(segment.text for segment in segments)
        # logger.debug(f"Transcription result: '{transcript.strip()}'")
        return transcript

    def language_detection(self, audio):
        self._load_model() # Ensure model is loaded
        # run language detection on the first 30 seconds of audio
        logger.info("Performing language detection...")
        info = self.model.transcribe(audio[: SAMPLING_RATE * 30])
        detected_lang = info.language
        logger.info(f"Detected language: {detected_lang} (Probability: {info.language_probability:.2f})")
        self.language = detected_lang
        return detected_lang

class OnlineASRProcessor:
    """ 
    Accepts audio chunks and processes them using FasterWhisperASR.
    Manages audio buffer and decides when to transcribe.
    MODIFIED to integrate directly with FasterWhisperASR and simplify.
    Original logic based on whisper_streaming BufferProcessor.
    """

    def __init__(self, asr: FasterWhisperASR, buffer_trimming=("segment", 1.5)):
        """
        Args:
          asr: Instance of FasterWhisperASR.
          buffer_trimming: Method and duration for trimming the audio buffer after transcription.
                           ("segment", 1.5) keeps 1.5 seconds after the last segment end.
                           ("sentence", 0) keeps nothing after the last full sentence.
        """
        self.asr = asr
        self.buffer_trimming_way, self.buffer_trimming_sec = buffer_trimming
        self.audio_buffer = np.array([], dtype=np.float32)
        self.prompt = "" # No initial prompt needed usually
        self.last_segment_end_ts: float = 0.0
        self.init()

    def init(self):
        """Initialize buffer and states."""
        logger.info("Initializing OnlineASRProcessor buffer and state.")
        self.audio_buffer = np.array([], dtype=np.float32)
        self.prompt = "" # Reset prompt on init
        self.last_segment_end_ts = 0.0

    def insert_audio_chunk(self, audio_chunk: np.ndarray):
        """Append audio chunk to the buffer."""
        self.audio_buffer = np.append(self.audio_buffer, audio_chunk)
        # logger.debug(f"Inserted {len(audio_chunk)/SAMPLING_RATE:.2f}s audio. Buffer size: {len(self.audio_buffer)/SAMPLING_RATE:.2f}s")

    def prompt_update(self, text: str):
        """Update the transcription prompt."""
        logger.debug(f"Updating prompt with: '{text.strip()}'")
        self.prompt = f"{self.prompt.strip()} {text.strip()}" # Append new text

    def _get_prompt(self) -> str:
        """Returns the current prompt."""
        # logger.debug(f"Using prompt: '{self.prompt.strip()}'")
        return self.prompt

    def process_iter(self) -> Optional[Tuple[float, float, str]]:
        """ 
        Transcribe the current audio buffer. Returns the last segment.
        Buffer trimming logic MODIFIED to work directly with faster-whisper segments.
        """
        if self.audio_buffer is None or len(self.audio_buffer) == 0:
            # logger.debug("process_iter called with empty buffer.")
            return None
        
        # logger.debug(f"Processing buffer of size: {len(self.audio_buffer)/SAMPLING_RATE:.2f}s")
        try:
            # Use faster-whisper's transcribe method, which returns segments
            segments, info = self.asr.model.transcribe(
                self.audio_buffer, 
                language=self.asr.language if self.asr.language != "auto" else None,
                initial_prompt=self._get_prompt(),
                # --- VAD Filtering (Optional but recommended for streaming) ---
                # These settings help filter out noise/silence segments directly.
                # Adjust thresholds as needed.
                vad_filter=True,
                vad_parameters=dict(threshold=0.5, min_silence_duration_ms=500)
            )

            # Convert generator to list to process segments
            segment_list = list(segments)

            if not segment_list:
                # logger.debug("No segments transcribed from the current buffer.")
                return None # No speech detected or transcribed

            # --- Process the last transcribed segment --- 
            last_segment = segment_list[-1]
            text = last_segment.text
            start_ts = last_segment.start
            end_ts = last_segment.end
            # logger.debug(f"Last segment: [{start_ts:.2f}s -> {end_ts:.2f}s] '{text.strip()}'")

            self.prompt_update(text)
            self.last_segment_end_ts = end_ts

            # --- Buffer Trimming --- 
            # Keeps audio after the specified point relative to the last segment end
            trim_point_seconds = self.last_segment_end_ts - self.buffer_trimming_sec
            trim_point_samples = int(trim_point_seconds * SAMPLING_RATE)
            
            if trim_point_samples < 0:
                trim_point_samples = 0 # Cannot trim before the start
            
            # Keep the audio data AFTER the trim point
            self.audio_buffer = self.audio_buffer[trim_point_samples:]
            # logger.debug(f"Buffer trimmed. Kept audio after {trim_point_seconds:.2f}s. New buffer size: {len(self.audio_buffer)/SAMPLING_RATE:.2f}s")

            # --- Adjust timestamps relative to the start of the *original* buffer processing --- 
            # This part is tricky. OnlineASRProcessor doesn't inherently know the absolute timeline.
            # The timestamps returned (start_ts, end_ts) are relative to the beginning of self.audio_buffer *before* transcription.
            # For a truly online system feeding chunks, these relative timestamps are often sufficient.
            # If absolute time is needed, the caller needs to manage it.
            # We return the relative timestamps here.
            return start_ts, end_ts, text

        except Exception as e:
            logger.error(f"Error during transcription in process_iter: {e}")
            # Don't clear buffer on error, maybe it's transient
            return None

    def finish(self) -> Optional[Tuple[float, float, str]]:
        """ Process the remaining audio buffer at the end of the stream. """
        logger.info("Processing final audio buffer...")
        final_result = self.process_iter() # Process whatever is left
        self.init() # Reset state after finishing
        if final_result:
            logger.info(f"Final segment: [{final_result[0]:.2f}s -> {final_result[1]:.2f}s] '{final_result[2].strip()}'")
        else:
            logger.info("No text transcribed from the final buffer.")
        return final_result 