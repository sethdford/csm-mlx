#!/usr/bin/env python
import argparse
import queue as std_queue
import sys
import time
from typing import List, Optional, Tuple, Callable, Generator
import re
import string
import asyncio
import concurrent.futures
import functools
from collections import Counter

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import sounddevice as sd
from huggingface_hub import hf_hub_download
from mlx_lm.sample_utils import make_sampler

# Import from csm-mlx library
from csm_mlx import CSM, csm_1b, Segment
from csm_mlx.generation import stream_generate

# --- Pipeline Components (Import necessary ones) ---
from faster_whisper import WhisperModel

# Import whisper_streaming utilities
from stt_processor import FasterWhisperASR, OnlineASRProcessor # Import from local file

# Import MLX LM for LLM
import mlx.core as mx
from mlx_lm import load, generate # Use main generate function
import mlx_lm

# Optional imports for context loading & saving output
import audiofile
import audresample
import soundfile as sf

# --- Configuration ---
DEFAULT_MODEL_REPO = "senstella/csm-1b-mlx"
DEFAULT_MODEL_FILE = "ckpt.safetensors"
DEFAULT_QUANTIZED_MODEL_FILE = "quantized_ckpt.safetensors"
DEFAULT_ADAPTER_FILE = "adapters.safetensors"
INPUT_SAMPLE_RATE = 16000
SAMPLE_RATE = 24000
AUDIO_CHUNK_SIZE = 1920
DEFAULT_SPEAKER_ID = 0
DEFAULT_TEMP = 0.6
DEFAULT_TOP_K = 50
DEFAULT_TOP_P = 1.0
DEFAULT_MIN_P = 0.0

# STT_MODEL_SIZE = "tiny.en"
# STT_DEVICE = "cpu"
# STT_COMPUTE_TYPE = "int8"
# VAD settings are less relevant now, OnlineASRProcessor handles segmentation
# VAD_THRESHOLD = 0.5
# VAD_MIN_SILENCE_DURATION_MS = 500
ONLINE_MIN_CHUNK_SECONDS = 0.2

# --- Pipeline Queues ---
audio_in_queue = std_queue.Queue()
stt_out_queue = std_queue.Queue()
llm_out_queue = std_queue.Queue()
audio_output_queue = std_queue.Queue()

# --- Control Events ---
stop_event = asyncio.Event()
tts_speaking_event = asyncio.Event()
interruption_event = asyncio.Event()

# --- Latency Tracking State ---
llm_response_start_time = None
last_response_latency = 0.0
LATENCY_THRESHOLD = 7.0
latency_lock = asyncio.Lock()

# --- Audio Output State ---
output_remainder = np.array([], dtype=np.float32)
playback_finished = asyncio.Event()
playback_truly_finished = asyncio.Event()

# List to store all generated chunks for saving later
all_generated_audio: List[np.ndarray] = []

# --- Configuration ---
LLM_MODEL_PATH = "mlx-community/Phi-3-mini-4k-instruct-4bit"
LLM_MAX_TOKENS = 200
LLM_TEMP = 0.7
LLM_PROMPT_TEMPLATE = "<|user|>\n{transcript}<|end|>\n<|assistant|>\n"

# Define silence duration in milliseconds
PAUSE_DURATION_MS = 50
SILENCE_CHUNK = np.zeros(int(SAMPLE_RATE * PAUSE_DURATION_MS / 1000), dtype=np.float32)

# Fade-out configuration
FADE_OUT_STEPS = 10 # Number of audio chunks to fade over

TTS_COOLDOWN_SECONDS = 0.4 # Add cooldown duration constant
MAX_CONTEXT_SEGMENTS = 6 # Define max context length

# Define sentinel object
LLM_RESPONSE_END = object()

# Add a length threshold
MIN_CHARS_FOR_INITIAL_TTS = 10 

# --- Central State Class ---
class ConversationState:
    def __init__(self, loop: asyncio.AbstractEventLoop):
        self.loop = loop

        # Queues
        self.audio_in_queue = asyncio.Queue(maxsize=200)
        self.stt_out_queue = asyncio.Queue(maxsize=50)
        self.llm_out_queue = asyncio.Queue(maxsize=50)
        self.sd_out_bridge_queue = std_queue.Queue(maxsize=1000)

        # Events
        self.stop_event = asyncio.Event()
        self.tts_speaking_event = asyncio.Event()
        self.interruption_event = asyncio.Event()
        self.tts_cooldown_active = asyncio.Event()
        self.playback_truly_finished = asyncio.Event()

        # Locks
        self.latency_lock = asyncio.Lock()
        self.fade_lock = asyncio.Lock()

        # Metrics / State Flags
        self.llm_response_start_time: Optional[float] = None
        self.last_response_latency: float = 0.0
        self.is_fading_out: bool = False
        self.fade_step: int = 0

        # Buffer for audio output callback
        self.output_remainder = np.array([], dtype=np.float32)

        # List to store all generated chunks for saving later
        self.all_generated_audio: List[np.ndarray] = []

        # Executor
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

    async def shutdown(self):
        print("[State] Shutting down executor...")
        self.executor.shutdown(wait=True, cancel_futures=True)
        print("[State] Executor shut down.")

    # --- Thread-Safe Methods for State Access (Examples - expand as needed) ---
    async def set_tts_speaking(self, speaking: bool):
        if speaking:
            if not self.tts_speaking_event.is_set(): # Only print on change
                print("[DEBUG STATE] TTS Speaking: True")
                self.tts_speaking_event.set()
        else:
            if self.tts_speaking_event.is_set(): # Only print on change
                print("[DEBUG STATE] TTS Speaking: False")
                self.tts_speaking_event.clear()

    def signal_interruption_threadsafe(self):
        if self.tts_speaking_event.is_set() and not self.interruption_event.is_set():
            print("[DEBUG STATE] Interruption Signaled")
            # Use call_soon_threadsafe to set the event from the callback thread
            self.loop.call_soon_threadsafe(self.interruption_event.set)
            # self.interruption_event.set() # Direct set might not be thread-safe depending on loop

    async def start_fade_out(self):
        async with self.fade_lock:
            if not self.interruption_event.is_set() or self.is_fading_out:
                return False
            print("[DEBUG STATE] Fade Out Started")
            self.is_fading_out = True
            self.fade_step = 0
            print("[DEBUG STATE] Clearing output bridge queue for fade-out...")
            cleared_count = 0
            while not self.sd_out_bridge_queue.empty():
                try: self.sd_out_bridge_queue.get_nowait(); cleared_count += 1
                except std_queue.Empty: break
            print(f"[DEBUG STATE] Cleared {cleared_count} items from output bridge queue.")
            return True

    async def get_fade_gain(self) -> Optional[float]:
        async with self.fade_lock:
            if not self.is_fading_out:
                return None
            gain = max(0.0, 1.0 - (self.fade_step / FADE_OUT_STEPS))
            # print(f"[DEBUG STATE] Fade Step: {self.fade_step}, Gain: {gain:.2f}") # Optional: can be noisy
            self.fade_step += 1
            if gain <= 0.0:
                print("[DEBUG STATE] Fade Gain Reached Zero")
            return gain

    async def finish_fade_out(self):
        async with self.fade_lock:
            # Check if we ARE fading before declaring finish
            was_fading = self.is_fading_out
            if not was_fading: return False # Indicate nothing was finished
            print("[DEBUG STATE] Fade Out Finished")
            self.is_fading_out = False
            self.interruption_event.clear()
            print("[DEBUG STATE] Interruption Event Cleared")
            return True # Indicate fade was finished

    async def record_llm_start(self):
        async with self.latency_lock:
            print("[DEBUG STATE] LLM Latency Timer Started")
            self.llm_response_start_time = time.time()

    async def record_llm_end(self):
        async with self.latency_lock:
            if self.llm_response_start_time is not None:
                self.last_response_latency = time.time() - self.llm_response_start_time
                print(f"[DEBUG STATE] LLM Latency Timer Stopped: {self.last_response_latency:.2f}s")
                self.llm_response_start_time = None
            else:
                print("[DEBUG STATE] Warning: Tried to stop LLM timer with no start time.")

    async def check_latency_threshold(self):
        async with self.latency_lock:
            if self.llm_response_start_time is None:
                return self.last_response_latency > LATENCY_THRESHOLD
            else:
                return False

# --- Audio Callbacks (Need access to loop and state) ---
_SHARED_STATE: Optional[ConversationState] = None

def audio_input_callback(indata, frames, time_info, status):
    """Sounddevice callback (runs in separate thread)."""
    global _SHARED_STATE
    if status:
        print(f"Audio Input Callback Status: {status}", file=sys.stderr)
    if _SHARED_STATE is None:
        print("[Input Callback Error] Shared state not initialized.", file=sys.stderr)
        raise sd.CallbackAbort
    
    state = _SHARED_STATE
    loop = state.loop

    if state.stop_event.is_set():
        raise sd.CallbackStop

    # Always queue audio if the stream is running.
    # The VAD/STT worker will handle discarding based on TTS state.
    try:
        # print("[DEBUG INPUT CB] Queuing audio chunk.") # Can be noisy
        # Use put_nowait for minimal blocking in the callback thread.
        loop.call_soon_threadsafe(state.audio_in_queue.put_nowait, indata.copy())
    except asyncio.QueueFull:
        # This is expected if the STT worker is paused (e.g., during TTS)
        # and the input buffer fills up. We just drop the data.
        # Reduce log level/frequency if this becomes too noisy.
        print("[Input Callback Debug] Audio input queue full (likely during TTS). Dropping chunk.", file=sys.stderr)
        pass # Discard the data
    except Exception as e:
        # Log other unexpected errors
        print(f"[Input Callback Error] Failed to queue audio: {e}", file=sys.stderr)

def audio_output_callback(outdata, frames, time, status):
    """Sounddevice callback (runs in separate thread)."""
    global _SHARED_STATE
    if status:
        pass
    if _SHARED_STATE is None:
        print("[Output Callback Error] Shared state not initialized.", file=sys.stderr)
        outdata.fill(0)
        raise sd.CallbackAbort

    state = _SHARED_STATE
    loop = state.loop

    if state.stop_event.is_set():
        outdata.fill(0)
        if not state.playback_truly_finished.is_set():
            loop.call_soon_threadsafe(state.playback_truly_finished.set)
        raise sd.CallbackStop

    try:
        needed = frames
        current_pos = 0

        if len(state.output_remainder) > 0:
            len_remainder = len(state.output_remainder)
            if len_remainder >= needed:
                outdata[:needed] = state.output_remainder[:needed].reshape(-1, 1)
                state.output_remainder = state.output_remainder[needed:]
                return
            else:
                outdata[:len_remainder] = state.output_remainder.reshape(-1, 1)
                current_pos = len_remainder
                needed -= len_remainder
                state.output_remainder = np.array([], dtype=np.float32)

        while needed > 0:
            try:
                data_chunk = state.sd_out_bridge_queue.get_nowait()
                # <<< Log Queue Get >>>
                qsize_after_get = state.sd_out_bridge_queue.qsize()
                chunk_len = len(data_chunk) if data_chunk is not None else 0
                print(f"[DEBUG Queue Get] Got chunk len: {chunk_len}, qsize after: {qsize_after_get}")
                # <<<

                if data_chunk is None:
                    print("[DEBUG OUTPUT] Received None sentinel.")
                    outdata[current_pos:].fill(0)
                    if not state.playback_truly_finished.is_set():
                        loop.call_soon_threadsafe(state.playback_truly_finished.set)
                    break

                len_chunk = len(data_chunk)
                if len_chunk >= needed:
                    outdata[current_pos : current_pos + needed] = data_chunk[:needed].reshape(-1, 1)
                    state.output_remainder = data_chunk[needed:]
                    needed = 0
                else:
                    outdata[current_pos : current_pos + len_chunk] = data_chunk.reshape(-1, 1)
                    needed -= len_chunk
                    current_pos += len_chunk
                    state.output_remainder = np.array([], dtype=np.float32)

            except std_queue.Empty:
                outdata[current_pos:].fill(0)
                state.output_remainder = np.array([], dtype=np.float32)
                # print("[DEBUG OUTPUT] Queue empty, filling with silence.") # Noisy
                break

    except sd.CallbackStop:
        if not state.playback_truly_finished.is_set():
            loop.call_soon_threadsafe(state.playback_truly_finished.set)
        raise
    except Exception as e:
        print(f"Error in output callback: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        outdata.fill(0)
        if not state.playback_truly_finished.is_set():
            loop.call_soon_threadsafe(state.playback_truly_finished.set)
        raise sd.CallbackAbort

# --- Async Worker Function Stubs / Implementations ---

def _stt_process_chunk(online_processor: OnlineASRProcessor, chunk: np.ndarray):
    """Runs insert and process_iter in sequence."""
    online_processor.insert_audio_chunk(chunk)
    return online_processor.process_iter()

def _stt_finish(online_processor: OnlineASRProcessor):
    """Runs finish."""
    return online_processor.finish()

async def vad_stt_worker(state: ConversationState, args: argparse.Namespace):
    """Async worker for VAD + STT using OnlineASRProcessor."""
    print("[STT Worker] Initializing (Async)...")
    loop = state.loop
    executor = state.executor
    # stt_model = None # Removed pre-loading of WhisperModel
    # print(f"Loading STT model (Faster Whisper {args.stt_model_size})...")
    # try:
    #     # Use functools.partial to correctly pass keyword arguments
    #     whisper_loader = functools.partial(
    #         WhisperModel, 
    #         args.stt_model_size, 
    #         device=args.stt_device, 
    #         compute_type=args.stt_compute_type
    #     )
    #     stt_model = await loop.run_in_executor(
    #         executor,
    #         whisper_loader
    #     )
    #     print("[STT Worker] Model loaded.")
    # except Exception as e:
    #     print(f"FATAL: Could not load Faster Whisper model: {e}", file=sys.stderr)
    #     state.stop_event.set()
    #     return

    online_processor: Optional[OnlineASRProcessor] = None
    def initialize_asr_processor():
        nonlocal online_processor
        try:
            print("[DEBUG STT] Initializing/Resetting OnlineASRProcessor...")
            # Pass args directly to FasterWhisperASR constructor
            # asr_backend = FasterWhisperASR(lan=args.stt_lang, model=stt_model) # Old way
            asr_backend = FasterWhisperASR(
                modelsize=args.stt_model_size, 
                lan=args.stt_lang, 
                device=args.stt_device, 
                compute_type=args.stt_compute_type
            )
            online_processor = OnlineASRProcessor(asr=asr_backend, buffer_trimming=("segment", 0.5))
            online_processor.init()
            print("[DEBUG STT] OnlineASRProcessor Ready.")
            return True
        except Exception as e:
            print(f"[STT Worker] Error initializing OnlineASRProcessor: {e}", file=sys.stderr)
            state.stop_event.set()
            return False

    if not initialize_asr_processor():
        return

    min_chunk_samples = int(args.online_min_chunk_seconds * INPUT_SAMPLE_RATE)
    awaiting_tts_finish = False # Flag to indicate we are waiting for TTS+cooldown to end

    print("[STT Worker] Ready. Processing audio based on TTS state...")
    try:
        while not state.stop_event.is_set():
            if online_processor is None:
                 print("[STT Worker Error] online_processor is None!", file=sys.stderr)
                 await asyncio.sleep(0.5)
                 continue

            is_tts_speaking = state.tts_speaking_event.is_set()
            is_tts_cooling_down = state.tts_cooldown_active.is_set()

            # --- Check if TTS is active or cooling down --- 
            if is_tts_speaking or is_tts_cooling_down:
                if not awaiting_tts_finish:
                    log_reason = "TTS started" if is_tts_speaking else "TTS cooldown started"
                    print(f"[DEBUG STT] {log_reason}. Halting ASR processing.")
                    awaiting_tts_finish = True
                # Keep halting while TTS is active or cooling down
                await asyncio.sleep(args.online_min_chunk_seconds / 2.0)
                continue
            
            # --- TTS is OFF and Cooldown is OFF --- 
            else:
                # --- If we *were* waiting for TTS to finish, it just did. Reset ASR. ---
                if awaiting_tts_finish:
                    print("[DEBUG STT] TTS finished (speaking & cooldown). Resetting ASR state and clearing buffer.")
                    if not initialize_asr_processor(): break
                    awaiting_tts_finish = False # Reset the flag
                    
                    # Discard audio that arrived during TTS/cooldown
                    discard_count = 0
                    while not state.audio_in_queue.empty():
                        try:
                            state.audio_in_queue.get_nowait()
                            state.audio_in_queue.task_done()
                            discard_count += 1
                        except asyncio.QueueEmpty:
                            break
                    print(f"[DEBUG STT] Discarded {discard_count} potentially stale audio chunks after TTS stop/cooldown.")
                    continue # Go back to the start of the loop to process fresh audio
                
                # --- Process Real Audio (TTS is off, cooldown is off, and we weren't waiting) --- 
                real_audio_chunk = None
                try:
                    real_audio_chunk = await asyncio.wait_for(state.audio_in_queue.get(), timeout=0.1)
                    state.audio_in_queue.task_done()
                except asyncio.TimeoutError:
                    continue
                except asyncio.QueueEmpty:
                    continue
                except Exception as e:
                    print(f"[STT Worker] Error getting audio chunk: {e}", file=sys.stderr)
                    await asyncio.sleep(0.05)
                    continue
                
                if real_audio_chunk is not None:
                    try:
                        o = await loop.run_in_executor(
                            executor, _stt_process_chunk, online_processor, real_audio_chunk
                        )
                        if o and o[2]:
                            processed_text = " " + o[2].strip()
                            if processed_text.strip():
                                print(f"[DEBUG STT] Committing text: '{processed_text.strip()}'")
                                await state.stt_out_queue.put(processed_text)
                    except Exception as e:
                        print(f"[STT Worker] Error during STT processing chunk: {e}", file=sys.stderr)

        # --- Shutdown processing ---
        print("[STT Worker] Stop event set or main loop exited.")
        # Process final buffer only if STT wasn't halted at the end
        if not awaiting_tts_finish and online_processor is not None:
            final_buffer = np.array([], dtype=np.float32)
            while not state.audio_in_queue.empty():
                try:
                    chunk = state.audio_in_queue.get_nowait()
                    final_buffer = np.append(final_buffer, chunk)
                    state.audio_in_queue.task_done()
                except asyncio.QueueEmpty:
                    break
            
            if len(final_buffer) > 0:
                print(f"[DEBUG STT] Processing final user audio buffer ({len(final_buffer)/INPUT_SAMPLE_RATE:.2f}s)...")
                try:
                    # Process the remaining buffer
                    o_iter = await loop.run_in_executor(executor, _stt_process_chunk, online_processor, final_buffer)
                    if o_iter and o_iter[2]:
                        final_text_iter = " " + o_iter[2].strip()
                        if final_text_iter.strip():
                            print(f"[DEBUG STT] Committing final iter text: '{final_text_iter.strip()}'")
                            await state.stt_out_queue.put(final_text_iter)
                    # Call finish to get any final segments
                    o_finish = await loop.run_in_executor(executor, _stt_finish, online_processor)
                    if o_finish and o_finish[2]: 
                        final_text_finish = " " + o_finish[2].strip()
                        if final_text_finish.strip():
                            print(f"[DEBUG STT] Committing final finish text: '{final_text_finish.strip()}'")
                            await state.stt_out_queue.put(final_text_finish)
                    print("[DEBUG STT] Final user buffer processed.")
                except Exception as e:
                    print(f"[STT Worker] Error during final STT processing: {e}", file=sys.stderr)
        else:
             print("[DEBUG STT] No final user audio buffer to process (TTS was active or processor unavailable).")

    except asyncio.CancelledError:
        print("[STT Worker] Task cancelled.")
    except Exception as e:
        print(f"[STT Worker] Unhandled error in main loop: {e}")
        import traceback
        traceback.print_exc()
        state.stop_event.set()
    finally:
        print("[STT Worker] Shutdown complete.")


# --- LLM Worker ---
async def llm_worker(state: ConversationState, args: argparse.Namespace):
    """Async worker for LLM processing."""
    print("[LLM Worker] Initializing (Async)...")
    loop = state.loop
    executor = state.executor
    print(f"Loading LLM model: {args.llm_model_path}")
    try:
        llm_model, llm_tokenizer = await loop.run_in_executor(
            executor, load, args.llm_model_path
        )
        print("[LLM Worker] LLM model loaded.")
    except Exception as e:
        print(f"FATAL: Could not load LLM model {args.llm_model_path}: {e}", file=sys.stderr)
        state.stop_event.set()
        return

    print("[LLM Worker] Ready. Waiting for transcripts...")
    accumulated_transcript = ""
    try:
        while not state.stop_event.is_set():
            try:
                # Wait for a transcript chunk with a timeout
                transcript_chunk = await asyncio.wait_for(state.stt_out_queue.get(), timeout=0.5)
                print("[DEBUG LLM] Dequeued item from stt_out_queue.")
                accumulated_transcript += transcript_chunk
                state.stt_out_queue.task_done()
                print(f"[DEBUG LLM] Received transcript chunk: '{transcript_chunk.strip()}'")
                print(f"[DEBUG LLM] Accumulated transcript: '{accumulated_transcript.strip()}'")

                # Check for sentence end to decide if we speak this part
                is_sentence_end = transcript_chunk.strip().endswith(tuple(string.punctuation + "\n"))
                if is_sentence_end or "\n" in transcript_chunk:
                    response_to_speak = accumulated_transcript.strip()
                    accumulated_transcript = "" # Reset for next turn
                    print(f"[DEBUG LLM] Processing transcript: '{response_to_speak}'")
                else:
                     # Not a potential sentence end, continue accumulating
                     continue

                # -- Only reach here if sentence end passed --

                # Format the prompt
                prompt = mx.array(llm_tokenizer.encode(LLM_PROMPT_TEMPLATE.format(transcript=response_to_speak)))

                await state.record_llm_start() # Record latency start

                response_buffer = ""
                skip = 0
                REPLACEMENT_CHAR = "\ufffd"
                CONTROL_TOKENS_TO_REMOVE = ["<|end|>", "<|user|>", "<|assistant|>", "<|eot_id|>", "<|start_header_id|>", "<|end_header_id|>", "assistant|>", "user|>"]
                END_TOKEN = "<|eot_id|>"
                ROLE_MARKERS_TO_TRUNCATE = ["user|>", "assistant|>", "<|end|>"]

                print("[DEBUG LLM] Starting generation via mlx_lm.generate...")
                token_iterator = generate(
                    llm_model,
                    llm_tokenizer,
                    prompt=LLM_PROMPT_TEMPLATE.format(transcript=response_to_speak),
                    max_tokens=args.llm_max_tokens,
                    verbose=False
                )

                generated_text = ""
                found_end_token = False
                # --- Generation Loop --- 
                for text_chunk in token_iterator:
                    if state.stop_event.is_set(): break
                    
                    processed_chunk = text_chunk.replace("\n\n", ". ")
                    
                    end_token_pos = processed_chunk.find(END_TOKEN)
                    if end_token_pos != -1:
                        chunk_before_end = processed_chunk[:end_token_pos]
                        generated_text += chunk_before_end
                        print(f"[DEBUG LLM] Detected {END_TOKEN}. Processing final part: '{chunk_before_end.strip()}'")
                        found_end_token = True
                        break
                    else:
                        generated_text += processed_chunk
                        generated_text = generated_text.replace(REPLACEMENT_CHAR, " ")
                    # No longer need to split/send words here, process full response below
                # --- End of Generation Loop --- 

                # --- Post-Generation Processing --- 
                print(f"[DEBUG LLM] Raw generated text (len={len(generated_text)}): '{generated_text[:200]}...'") # Log raw output
                
                # 1. Truncate at first role marker (if any)
                final_response = generated_text
                truncate_pos = -1
                for marker in ROLE_MARKERS_TO_TRUNCATE:
                    pos = final_response.find(marker)
                    if pos != -1:
                        if truncate_pos == -1 or pos < truncate_pos:
                            truncate_pos = pos
                            
                if truncate_pos != -1:
                    print(f"[DEBUG LLM] Truncating response at position {truncate_pos} due to marker.")
                    final_response = final_response[:truncate_pos]

                # 2. Filter remaining control tokens and clean up
                for token in CONTROL_TOKENS_TO_REMOVE:
                    final_response = final_response.replace(token, "")
                final_response = final_response.strip()

                # 3. Filter for Garbage Output
                if is_likely_garbage(final_response):
                    print(f"[DEBUG LLM] Replacing likely garbage output: '{final_response[:100]}...'")
                    final_response = "I seem to be having trouble processing that." # Fallback message

                # 4. Send the processed response (potentially empty)
                if final_response:
                    print(f"[DEBUG LLM] Queuing final processed response: '{final_response}'")
                    # Send the whole response as one item for TTS to handle sentence splitting
                    await state.llm_out_queue.put(final_response)
                else:
                    print("[DEBUG LLM] Response is empty after filtering/truncation.")
                    
                # <<< Send End Sentinel >>>
                print("[DEBUG LLM] Queuing LLM_RESPONSE_END sentinel.")
                await state.llm_out_queue.put(LLM_RESPONSE_END)
                # <<< End Sentinel >>>

                await state.record_llm_end()
                print(f"[DEBUG LLM] Finished processing transcript.")

            except asyncio.TimeoutError:
                # No transcript chunk received, continue loop
                continue
            except asyncio.QueueEmpty:
                # Should ideally not happen with timeout, but handle defensively
                await asyncio.sleep(0.05)
                continue

    except asyncio.CancelledError:
        print("[LLM Worker] Cancelled.")
    except Exception as e:
        print(f"[LLM Worker] Error in loop: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        state.stop_event.set()
    finally:
        print("[LLM Worker] Shutdown.")

# --- Helper Function to Detect Garbage --- 
def is_likely_garbage(text: str, threshold: float = 0.30) -> bool:
    """Checks if the text is likely garbage based on non-alphanumeric ratio and word repetition."""
    if not text or text.isspace():
        return False # Empty or whitespace is not garbage in this context

    # 1. Character Ratio Check (existing)
    allowed_punctuation = string.punctuation + " "
    non_alnum_count = 0
    total_chars = 0
    clean_text_for_words = '' # Build clean text for word analysis

    for char in text:
        total_chars += 1
        is_allowed_punct = char in allowed_punctuation
        is_alnum = char.isalnum()
        if not is_alnum and not is_allowed_punct:
            non_alnum_count += 1
        # Add letters, numbers, and spaces to the clean text
        if is_alnum or char == ' ':
            clean_text_for_words += char

    if total_chars == 0: return False # Avoid division by zero

    char_ratio = non_alnum_count / total_chars
    if char_ratio > threshold:
        print(f"[DEBUG LLM Filter] Detected garbage (Char Ratio). Ratio: {char_ratio:.2f} > {threshold:.2f}")
        return True

    # 2. Character Repetition Check (existing)
    if total_chars > 10:
        for char_to_check in set(c for c in text if not c.isspace()):
             if text.count(char_to_check * 5) > 0: # Check for 5+ repetitions
                  print(f"[DEBUG LLM Filter] Detected garbage (Char Repetition of '{char_to_check}')")
                  return True

    # 3. Word Repetition & Variety Check (New)
    words = clean_text_for_words.lower().split()
    total_words = len(words)

    if total_words > 5: # Only apply word checks if there are enough words
        word_counts = Counter(words)
        most_common_word, most_common_count = word_counts.most_common(1)[0]
        unique_word_count = len(word_counts)

        # Check if the most common word makes up > 40% of all words
        if most_common_count / total_words > 0.40:
            print(f"[DEBUG LLM Filter] Detected garbage (Word Repetition '{most_common_word}'). Count: {most_common_count}/{total_words}")
            return True
            
        # Check if the number of unique words is very low (<25% of total words)
        if unique_word_count / total_words < 0.25 and total_words > 10: # Add length check to avoid penalizing short valid phrases
            print(f"[DEBUG LLM Filter] Detected garbage (Low Word Variety). Unique: {unique_word_count}/{total_words}")
            return True

    return False # If none of the checks triggered

# --- Helper Functions (Remain the same) ---
def find_sentence_end(buffer: str) -> int:
    # Updated regex to better handle sentence endings potentially followed by spaces or end of string ($).
    # It looks for '.', '?', '!', or '\n' followed by whitespace or the end of the string ($).
    match = re.search(r"[.?!](\s+|$)|[\n]", buffer)
    if match:
        # Return the position *after* the punctuation and any trailing whitespace
        # This helps ensure we capture the full sentence ending pattern.
        return match.end()
    return -1

def load_and_apply_adapters(model: CSM, adapter_file: str):
    print(f"Loading adapter weights from: {adapter_file}")
    try:
        adapters = mx.load(adapter_file)
        model.load_weights(list(adapters.items()), strict=False)
        mx.eval(model.parameters())
        print("Adapter weights loaded and applied successfully.")
    except Exception as e:
        print(f"Error loading or applying adapter weights from {adapter_file}: {e}", file=sys.stderr)
        print("Continuing with base model only.")

def list_audio_devices():
    print("Available audio devices:")
    try:
        devices = sd.query_devices()
        print(devices)
    except Exception as e:
        print(f"Could not query audio devices: {e}", file=sys.stderr)

def read_audio_for_context(audio_path: str, sampling_rate: int = SAMPLE_RATE) -> Optional[mx.array]:
    try:
        print(f"  Loading context audio: {audio_path}")
        signal, original_sampling_rate = audiofile.read(audio_path, always_2d=True)
        
        if original_sampling_rate != sampling_rate:
            print(f"    Resampling context audio from {original_sampling_rate}Hz to {sampling_rate}Hz")
            signal_resampled = audresample.resample(signal, original_sampling_rate, sampling_rate)
        else:
            signal_resampled = signal

        signal_mx = mx.array(signal_resampled)
        if signal_mx.ndim > 1 and signal_mx.shape[1] > 1:
            print("    Converting context audio to mono.")
            signal_mx = signal_mx.mean(axis=1)
        else:
            signal_mx = signal_mx.squeeze()

        mx.eval(signal_mx)
        return signal_mx
    except Exception as e:
        print(f"  Error reading context audio {audio_path}: {e}", file=sys.stderr)
        return None

def load_model_and_weights(args: argparse.Namespace) -> CSM:
    print("Loading CSM model configuration...")
    csm_config = csm_1b()
    model = CSM(csm_config)

    weights_loaded = False
    quantized_applied = False
    weights_filename = None

    print(f"Loading weights from Hugging Face Hub: {args.model_repo}...")
    if args.quantize:
        try:
            weights_filename = DEFAULT_QUANTIZED_MODEL_FILE
            print(f"Attempting to load pre-quantized weights: {weights_filename}")
            weights_path = hf_hub_download(repo_id=args.model_repo, filename=weights_filename)
            model.load_weights(weights_path)
            print("Loaded pre-quantized weights successfully.")
            weights_loaded = True
            quantized_applied = True
        except Exception as e:
            print(f"Could not find/load pre-quantized weights: {e}. Will try default weights and quantize later.")

    if not weights_loaded:
        try:
            weights_filename = DEFAULT_MODEL_FILE
            print(f"Loading default weights: {weights_filename}")
            weights_path = hf_hub_download(repo_id=args.model_repo, filename=weights_filename)
            model.load_weights(weights_path)
            print("Loaded default weights successfully.")
            weights_loaded = True
        except Exception as e_fallback:
            print(f"FATAL Error: Could not load any model weights ('{DEFAULT_QUANTIZED_MODEL_FILE}' or '{DEFAULT_MODEL_FILE}') from {args.model_repo}: {e_fallback}", file=sys.stderr)
            sys.exit(1)

    if args.quantize and not quantized_applied:
        print("Applying quantization...")
        try:
            nn.quantize(model, args.quantize_group_size, args.quantize_bits)
            print(f"Model quantized with group size {args.quantize_group_size} and {args.quantize_bits} bits.")
            quantized_applied = True
        except Exception as e:
            print(f"Warning: Failed to apply quantization: {e}", file=sys.stderr)
    elif args.quantize and quantized_applied:
        print("Using pre-quantized weights.")
    elif not args.quantize:
        print("Quantization not requested. Using full precision weights.")

    model.eval()
    mx.eval(model.parameters())
    print("Model ready.")
    return model

# --- Function to generate synchronously --- 
def _generate_sync(
    model: CSM, # Changed back to CSM
    text: str,
    context: list[Segment],
    speaker: int,
    temperature: float, # Added temperature arg
    max_audio_length_ms: float = 90_000,
    logits_processors: Optional[List[Callable[[mx.array, mx.array], mx.array]]] = None,
) -> Generator[np.ndarray, None, None]:
    """Synchronously generates audio chunks by iterating the stream_generate."""
    sync_audio_generator = None # Initialize
    try:
        # stream_generate IS the synchronous generator
        print("[DEBUG Sync Gen] Calling stream_generate...") # Add log
        sync_audio_generator = stream_generate(
            model=model,
            text=text,
            temperature=temperature,
            context=context,
            speaker=speaker,
            max_audio_length_ms=max_audio_length_ms,
            logits_processors=logits_processors,
        )
        print("[DEBUG Sync Gen] stream_generate returned. Starting iteration...") # Add log

        # Iterate through the chunks yielded by stream_generate
        chunk_index = 0
        for audio_chunk_mx in sync_audio_generator:
            # print(f"[DEBUG Sync Gen] Yielding chunk {chunk_index}") # Optional: very noisy
            if audio_chunk_mx is not None and audio_chunk_mx.size > 0:
                # Convert to numpy for the main async loop
                yield np.array(audio_chunk_mx, dtype=np.float32)
            else:
                print(f"[DEBUG Sync Gen] Received None or empty chunk {chunk_index} from generator.")
            chunk_index += 1
        print(f"[DEBUG Sync Gen] Finished iteration after {chunk_index} chunks.") # Add log

    except Exception as e:
        print(f"[Sync Generator Helper] Error during generation/iteration: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc() # Print full traceback
    finally:
        print("[DEBUG Sync Gen] Exiting _generate_sync.") # Add log

# --- TTS Worker ---
async def tts_worker(state: ConversationState, args: argparse.Namespace, tts_model: CSM): # Removed sampler arg
    """Async worker for TTS generation and output queuing."""
    print("[TTS Worker] Initializing (Async)...")
    loop = state.loop
    executor = state.executor
    current_context: List[Segment] = []
    context_audio: List[mx.array] = []

    # Load initial context if provided
    if args.context_audio:
        print("[TTS Worker] Loading initial context audio...")
        for i, audio_path in enumerate(args.context_audio):
            audio_mx = await loop.run_in_executor(executor, read_audio_for_context, audio_path)
            if audio_mx is not None:
                text = args.context_text[i] if args.context_text and i < len(args.context_text) else ""
                spk = args.context_speaker[i] if args.context_speaker and i < len(args.context_speaker) else args.speaker
                current_context.append(Segment(text, int(spk), audio_mx)) # Use current_context
                context_audio.append(audio_mx)
        if current_context:
            print(f"[TTS Worker] Loaded {len(current_context)} initial context segments.")

    print("[TTS Worker] Ready. Waiting for LLM output...")

    try:
        while not state.stop_event.is_set():
            llm_output_item = None
            try:
                # Wait indefinitely for an item from the LLM worker
                llm_output_item = await state.llm_out_queue.get()
                print(f"[DEBUG TTS] Dequeued item type from llm_out_queue: {type(llm_output_item)}")
            except asyncio.CancelledError:
                print("[TTS Worker] Queue get cancelled.")
                break # Exit loop if cancelled
            except Exception as q_err:
                print(f"[TTS Worker Error] Error getting from llm_out_queue: {q_err}")
                await asyncio.sleep(0.1) # Avoid busy-looping on error
                continue

            # --- Process Dequeued Item --- 

            if isinstance(llm_output_item, str):
                state.llm_out_queue.task_done() # Mark task done for the string item
                full_response_text = llm_output_item
                print(f"[DEBUG TTS] Received full LLM response text: '{full_response_text}'")

                # --- Split into Segments ---
                segments_to_speak = re.split(r'([.?!]\s+|\n)', full_response_text)
                processed_segments = []
                if segments_to_speak:
                    current_segment = segments_to_speak[0]
                    for i in range(1, len(segments_to_speak), 2):
                         if i + 1 < len(segments_to_speak):
                              current_segment += segments_to_speak[i]
                              if segments_to_speak[i+1]: # Only add if next part exists
                                   processed_segments.append(current_segment.strip())
                                   current_segment = segments_to_speak[i+1]
                         else: # Handle the last punctuation part
                             current_segment += segments_to_speak[i]
                    # Add the final remaining part if it's not empty
                    if current_segment and current_segment.strip():
                        processed_segments.append(current_segment.strip())
                # Filter out any potentially empty segments again after stripping
                processed_segments = [s for s in processed_segments if s]

                print(f"[DEBUG TTS] Split into {len(processed_segments)} segments: {processed_segments}")

                # --- Process Each Segment --- 
                segment_processing_failed = False
                for response_to_speak in processed_segments:
                    if not response_to_speak: continue # Should be filtered, but double-check
                    if state.stop_event.is_set():
                        segment_processing_failed = True
                        break

                    print(f"[DEBUG TTS] Processing segment: '{response_to_speak}'")

                    # --- Skip Short Initial Greetings --- 
                    is_first_turn = not current_context
                    is_short_greeting = len(response_to_speak) <= MIN_CHARS_FOR_INITIAL_TTS
                    if is_first_turn and is_short_greeting:
                        print(f"[DEBUG TTS] Skipping TTS for short initial segment: '{response_to_speak}'")
                        silent_audio_len = 500 # Arbitrary short silence length
                        silent_audio_mx = mx.zeros((silent_audio_len,), dtype=mx.float32)
                        mx.eval(silent_audio_mx)
                        try:
                            new_segment = Segment(text=response_to_speak, speaker=args.speaker, audio=silent_audio_mx)
                            current_context.append(new_segment)
                            print(f"[DEBUG TTS Context] Added DUMMY segment. Context size: {len(current_context)}")
                            if len(current_context) > MAX_CONTEXT_SEGMENTS:
                                num_to_remove = len(current_context) - MAX_CONTEXT_SEGMENTS
                                current_context = current_context[num_to_remove:]
                                print(f"[DEBUG TTS Context] Context trimmed to {len(current_context)} segments.")
                        except Exception as context_e:
                            print(f"[TTS Worker Error] Failed to update context with DUMMY segment: {context_e}", file=sys.stderr)
                        continue # Skip to the next segment
                    # --- End Skip --- 

                    print(f"[DEBUG TTS] Starting generation for segment: '{response_to_speak}'")
                    if not state.tts_speaking_event.is_set():
                        await state.set_tts_speaking(True)
                    state.interruption_event.clear()
                    state.playback_truly_finished.clear() # Reset for this segment/utterance

                    segment_audio_chunks = []
                    generation_successful = True
                    text_to_generate = response_to_speak

                    try:
                        print(f"[DEBUG TTS Gen Call] Text: '{text_to_generate}', Context Segments: {len(current_context)}")
                        sync_generator_object = await loop.run_in_executor(
                            executor, _generate_sync, tts_model, text_to_generate, current_context, args.speaker, args.temperature
                        )

                        get_next_chunk = functools.partial(next, sync_generator_object, None)

                        print("[DEBUG TTS] Starting iteration over sync generator via executor...")
                        segment_start_time = time.time()
                        MAX_SEGMENT_GENERATION_TIME = 30.0

                        while not state.stop_event.is_set():
                            if time.time() - segment_start_time > MAX_SEGMENT_GENERATION_TIME:
                                print(f"[TTS Worker Error] Generation timeout (> {MAX_SEGMENT_GENERATION_TIME}s) for segment: '{text_to_generate[:50]}...'", file=sys.stderr)
                                generation_successful = False
                                break

                            audio_chunk_np = await loop.run_in_executor(executor, get_next_chunk)

                            if audio_chunk_np is None:
                                print("[DEBUG TTS] Generator finished.")
                                break # Finished generating for this segment

                            if audio_chunk_np.size == 0:
                                print("[DEBUG Chunk Stats] Received empty chunk")
                                continue # Skip empty chunks

                            gain = await state.get_fade_gain()
                            chunk_to_put = audio_chunk_np

                            if gain is not None:
                                chunk_to_put = audio_chunk_np * gain
                                if gain <= 0.0:
                                    print("[DEBUG TTS] Fade complete during segment generation.")
                                    await state.finish_fade_out()
                                    generation_successful = False
                                    break
                            elif generation_successful:
                                 segment_audio_chunks.append(audio_chunk_np)

                            try:
                                await loop.run_in_executor(
                                    executor,
                                    state.sd_out_bridge_queue.put,
                                    chunk_to_put,
                                    True, 0.5
                                )
                            except std_queue.Full:
                                print("[TTS Worker Warning] Sounddevice output queue full during segment (timeout exceeded). Discarding chunk.", file=sys.stderr)
                                if await state.get_fade_gain() is not None: await state.finish_fade_out()
                                generation_successful = False
                                break
                            except Exception as put_e:
                                print(f"[TTS Worker Error] Unexpected error putting chunk: {put_e}", file=sys.stderr)
                                generation_successful = False
                                break

                        # --- After generator loop for one segment ---
                        if not generation_successful or state.stop_event.is_set():
                           segment_processing_failed = True # Mark failure for the whole response
                           break # Break the outer FOR loop (over segments)

                        # Add inter-segment silence if successful
                        if generation_successful:
                            try:
                                await loop.run_in_executor(
                                    executor,
                                    state.sd_out_bridge_queue.put,
                                    SILENCE_CHUNK, True, 0.5
                                )
                            except std_queue.Full:
                                print("[TTS Worker Warning] Output queue full adding inter-segment silence (timeout exceeded).", file=sys.stderr)
                                generation_successful = False
                                segment_processing_failed = True # Mark failure
                                break # Break the outer FOR loop

                        # Update context if segment + silence were successful
                        if generation_successful and segment_audio_chunks:
                            try:
                                print("[DEBUG TTS Context] Segment successful, updating context.")
                                full_utterance_np = np.concatenate(segment_audio_chunks)
                                if full_utterance_np.dtype != np.float32: full_utterance_np = full_utterance_np.astype(np.float32)
                                full_utterance_mx = mx.array(full_utterance_np)
                                mx.eval(full_utterance_mx)
                                new_segment = Segment(text=response_to_speak, speaker=args.speaker, audio=full_utterance_mx)
                                current_context.append(new_segment)
                                print(f"[DEBUG TTS Context] Added segment. Context size: {len(current_context)}")
                                if len(current_context) > MAX_CONTEXT_SEGMENTS:
                                    num_to_remove = len(current_context) - MAX_CONTEXT_SEGMENTS
                                    current_context = current_context[num_to_remove:]
                                    print(f"[DEBUG TTS Context] Context trimmed to {len(current_context)} segments.")
                            except Exception as context_e:
                                 print(f"[TTS Worker Error] Failed to update context after segment: {context_e}", file=sys.stderr)
                        elif not generation_successful:
                             print("[DEBUG TTS Context] Segment generation failed or timed out, skipping context update.")

                    except Exception as gen_e: # Catch errors during the generation call itself
                        print(f"[DEBUG TTS Context] Segment generation exception: {gen_e}, skipping context update.")
                        import traceback; traceback.print_exc()
                        generation_successful = False
                        segment_processing_failed = True # Mark failure
                        break # Break the outer FOR loop

                # --- After looping through all segments for a string response --- 
                if segment_processing_failed:
                    print("[DEBUG TTS] Skipping end-of-response processing due to failure or stop during segments.")
                    # Ensure TTS state is cleaned up if we broke mid-response
                    if state.tts_speaking_event.is_set():
                        await state.finish_fade_out()
                        await state.set_tts_speaking(False)
                    if state.tts_cooldown_active.is_set():
                        state.tts_cooldown_active.clear()
                    # Continue to the next iteration of the main WHILE loop to get next item
                    continue 
                
                # If all segments processed successfully, the loop finishes.
                # We simply continue to the next iteration of the main WHILE loop,
                # expecting the LLM_RESPONSE_END sentinel next.
                print("[DEBUG TTS] Finished processing all segments for the current response text.")

            # --- Handle Sentinel ---             
            elif llm_output_item == LLM_RESPONSE_END:
                state.llm_out_queue.task_done() # Mark task done for the sentinel item
                print("[DEBUG TTS] Received LLM_RESPONSE_END sentinel.")
                
                # Check if TTS was speaking OR if the queue still has data (e.g., trailing silence)
                if state.tts_speaking_event.is_set() or not state.sd_out_bridge_queue.empty():
                   print("[DEBUG TTS] LLM_RESPONSE_END: Triggering final wait and cooldown.")
                   try:
                       await loop.run_in_executor(
                           executor,
                           state.sd_out_bridge_queue.put,
                           None,  # The sentinel
                           True,  # block=True
                           1.0    # timeout=1.0 seconds (longer for final wait)
                       )
                       print("[DEBUG TTS] Utterance sentinel sent to output queue (End of LLM response).")
                   except std_queue.Full:
                       print("[TTS Worker Warning] Output queue full for utterance sentinel (End of LLM response, timeout).", file=sys.stderr)

                   print("[TTS Worker] Waiting for final utterance playback completion...")
                   try:
                       await asyncio.wait_for(state.playback_truly_finished.wait(), timeout=15.0)
                       print("[TTS Worker] Final utterance playback complete.")
                   except asyncio.TimeoutError:
                       print("[TTS Worker Warning] Timeout waiting for final utterance playback completion signal.")

                   # Cleanup Fade/Interruption State
                   interrupted = state.interruption_event.is_set()
                   fade_finished = await state.finish_fade_out()
                   if interrupted and not fade_finished:
                       print("[DEBUG TTS] Interruption was set but fade wasn't active/finished. Clearing interruption event.")
                       state.interruption_event.clear()
                   elif not interrupted and fade_finished: print("[DEBUG TTS] Fade finished normally post-playback.")
                   elif interrupted and fade_finished: print("[DEBUG TTS] Fade finished due to interruption post-playback.")
                   
                   await state.set_tts_speaking(False) # Clear speaking state *after* cleanup
                   print("[DEBUG TTS] Finished full LLM response processing & playback.")

                   print(f"[DEBUG TTS] Starting {TTS_COOLDOWN_SECONDS}s cooldown post-response...")
                   state.tts_cooldown_active.set()
                   await asyncio.sleep(TTS_COOLDOWN_SECONDS)
                   state.tts_cooldown_active.clear()
                   print("[DEBUG TTS] Cooldown finished.")
                else:
                    # Handle case where sentinel received but TTS wasn't actually speaking
                    print("[DEBUG TTS] LLM_RESPONSE_END received, but no pending TTS actions. Ensuring TTS state is False.")
                    if state.tts_speaking_event.is_set(): await state.set_tts_speaking(False)
                    if state.tts_cooldown_active.is_set(): state.tts_cooldown_active.clear()
            
            # --- Handle Unexpected Item --- 
            else:
                print(f"[TTS Worker Warning] Received unexpected item type from llm_out_queue: {type(llm_output_item)}")
                try:
                    state.llm_out_queue.task_done() # Mark task done if possible
                except Exception:
                    pass
                # Continue to the next iteration of the main WHILE loop

    except asyncio.CancelledError:
        print("[TTS Worker] Cancelled.")
    except Exception as e:
        print(f"[TTS Worker] Error in main loop: {e}", file=sys.stderr)
        import traceback; traceback.print_exc()
        state.stop_event.set()
    finally:
        # --- Final Shutdown Cleanup ---
        state.tts_cooldown_active.clear() # Ensure cooldown is off
        try:
            # Use blocking put with timeout via executor for the FINAL sentinel
            await loop.run_in_executor(
                executor,
                state.sd_out_bridge_queue.put,
                None,  # The sentinel
                True,  # block=True
                0.5    # timeout=0.5 seconds (shorter for shutdown)
            )
            print("[DEBUG TTS] Final shutdown sentinel sent to output queue.")
        except std_queue.Full:
            print("[TTS Worker Warning] Output bridge queue full on final shutdown sentinel (timeout).", file=sys.stderr)
        except Exception as final_put_e:
             print(f"[TTS Worker Warning] Error sending final shutdown sentinel: {final_put_e}", file=sys.stderr)

        print("[TTS Worker] Final wait for playback completion...")
        try:
            # Wait for playback, but with a shorter timeout during shutdown
            await asyncio.wait_for(state.playback_truly_finished.wait(), timeout=3.0)
            print("[TTS Worker] Final playback deemed complete on shutdown.")
        except asyncio.TimeoutError:
            print("[TTS Worker Warning] Timeout waiting for final playback completion signal during shutdown.")
        finally:
             # Ensure fade/speaking state is cleared regardless of playback signal
             await state.finish_fade_out() # Attempt final fade clear
             await state.set_tts_speaking(False)
             print("[TTS Worker] Shutdown complete.")

# --- Main Async Function ---
async def main_async(args: argparse.Namespace):
    global _SHARED_STATE
    loop = asyncio.get_running_loop()
    state = ConversationState(loop)
    _SHARED_STATE = state

    # --- Force disable quantization for debugging audio issues --- 
    # if args.quantize:
    #     print("[DEBUG MAIN] Forcing quantization OFF for this run.")
    #     args.quantize = False
    # ------------------------------------------------------------

    print(f"Setting up streaming pipelines with MLX...")
    try:
        # Load and prepare the TTS model
        tts_model = load_model_and_weights(args)
        
        # Apply adapters if provided
        adapter_loaded_successfully = False
        if args.adapter_file:
            load_and_apply_adapters(tts_model, args.adapter_file)
            # We assume success if no exception was raised in the function
            # A more robust check might involve checking model parameters
            adapter_loaded_successfully = True # Assume success for now
        
        # Add confirmation log
        if adapter_loaded_successfully:
            print("[Main] Adapter weights confirmed loaded.")
        elif args.adapter_file:
            print("[Main] Adapter file specified, but loading might have failed (check logs above). Proceeding without adapter.")
        else:
            print("[Main] No adapter file specified. Using base model.")
            
        # Start worker tasks
        vad_stt_task = asyncio.create_task(vad_stt_worker(state, args))
        llm_task = asyncio.create_task(llm_worker(state, args))
        tts_task = asyncio.create_task(tts_worker(state, args, tts_model)) # Removed sampler from call
        
        # Setup audio streams
        input_device = args.input_device
        output_device = args.output_device
        
        # Open the audio streams
        input_stream = sd.InputStream(
            device=input_device,
            samplerate=INPUT_SAMPLE_RATE,
            channels=1,
            blocksize=AUDIO_CHUNK_SIZE,
            callback=audio_input_callback,
        )
        
        output_stream = sd.OutputStream(
            device=output_device,
            samplerate=SAMPLE_RATE,
            channels=1,
            blocksize=AUDIO_CHUNK_SIZE * 3,
            callback=audio_output_callback,
        )
        
        # Start the audio streams
        with input_stream, output_stream:
            print("Audio streams opened. Press Ctrl+C to stop...")
            # Keep the main task alive until interrupted or stopped
            while not state.stop_event.is_set():
                await asyncio.sleep(0.1)
    
    except asyncio.CancelledError:
        print("Main task cancelled.")
    except KeyboardInterrupt:
        print("Keyboard interrupt received in main task.")
    except Exception as e:
        print(f"Error in main task: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        state.stop_event.set()
        await asyncio.sleep(0.5)  # Allow workers to notice stop event
        
        # Wait for tasks to complete
        tasks_pending = []
        if 'vad_stt_task' in locals() and not vad_stt_task.done():
            tasks_pending.append(vad_stt_task)
        if 'llm_task' in locals() and not llm_task.done():
            tasks_pending.append(llm_task)
        if 'tts_task' in locals() and not tts_task.done():
            tasks_pending.append(tts_task)
            
        if tasks_pending:
            print(f"Waiting for {len(tasks_pending)} tasks to complete...")
            await asyncio.gather(*tasks_pending, return_exceptions=True)
            
        # Final cleanup
        await state.shutdown()

        # Save generated audio if requested
        if args.output_file and state.all_generated_audio:
            print(f"\nSaving generated audio to {args.output_file}...")
            try:
                full_audio = np.concatenate(state.all_generated_audio)
                # Use soundfile to save as WAV
                await loop.run_in_executor(
                    None, # Use default executor for potentially blocking I/O
                    sf.write, args.output_file, full_audio, SAMPLE_RATE
                )
                print(f"Audio saved successfully.")
            except Exception as save_e:
                print(f"Error saving audio file: {save_e}", file=sys.stderr)

        print("Main task complete. Exiting.")

# --- Entry Point ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Async Streaming Voice-to-Voice Conversation",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-i", "--input-device", type=int, help="Input audio device ID (optional, uses default if not specified)")
    parser.add_argument("-o", "--output-device", type=int, help="Output audio device ID (optional, uses default if not specified)")
    parser.add_argument("--output-file", type=str, help="Path to save the generated audio as a WAV file (optional)")
    parser.add_argument("--list-devices", action="store_true", help="List available audio devices and exit")

    model_group = parser.add_argument_group('TTS Model and Quantization')
    model_group.add_argument("--model-repo", type=str, default=DEFAULT_MODEL_REPO, help="Hugging Face repository ID for the CSM TTS model")
    model_group.add_argument("--quantize", action="store_true", help="Apply MLX quantization to the TTS model")
    model_group.add_argument("--quantize-group-size", type=int, default=64, help="Group size for TTS quantization")
    model_group.add_argument("--quantize-bits", type=int, default=4, help="Bits for TTS quantization")
    model_group.add_argument("--adapter-file", type=str, default=None, help="Path to the local adapter file (e.g., adapters.safetensors) to apply over the base TTS model.")

    stt_group = parser.add_argument_group('STT Parameters')
    stt_group.add_argument("--stt-model-size", type=str, default="tiny.en", help="Faster Whisper model size (tiny.en, base.en, small.en, etc.)")
    stt_group.add_argument("--stt-device", type=str, default="cpu", help="Device for STT model (cpu, cuda, mps)")
    stt_group.add_argument("--stt-compute-type", type=str, default="int8", help="Compute type for STT model (int8, float16, etc.)")
    stt_group.add_argument("--stt-lang", type=str, default="en", help="Language code for STT model (e.g., en, de, auto)")
    stt_group.add_argument("--online-min-chunk-seconds", type=float, default=0.2, help="Minimum chunk size in seconds for whisper_streaming processor")

    gen_group = parser.add_argument_group('TTS Generation Parameters')
    gen_group.add_argument("-s", "--speaker", type=int, default=DEFAULT_SPEAKER_ID, help="Speaker ID for TTS output")
    gen_group.add_argument("-t", "--temperature", type=float, default=0.6, help="Sampling temperature for TTS")
    gen_group.add_argument("-k", "--top-k", type=int, default=DEFAULT_TOP_K, help="Sampling top-k for TTS")
    gen_group.add_argument("--top-p", type=float, default=DEFAULT_TOP_P, help="Sampling top-p (nucleus sampling) for TTS")
    gen_group.add_argument("--min-p", type=float, default=DEFAULT_MIN_P, help="Minimum probability for sampling for TTS")

    context_group = parser.add_argument_group('Initial Context Parameters (Optional)')
    context_group.add_argument("--context-audio", type=str, nargs='*', help="List of initial context audio file paths")
    context_group.add_argument("--context-text", type=str, nargs='*', help="List of initial context text transcripts")
    context_group.add_argument("--context-speaker", type=str, nargs='*', help="List of initial context speaker IDs")

    llm_group = parser.add_argument_group('LLM Parameters')
    llm_group.add_argument("--llm-model-path", type=str, default=LLM_MODEL_PATH, help="Path or HF repo ID for the MLX LLM model")
    llm_group.add_argument("--llm-max-tokens", type=int, default=LLM_MAX_TOKENS, help="Max generation tokens for LLM")
    llm_group.add_argument("--llm-temp", type=float, default=LLM_TEMP, help="Sampling temperature for LLM")

    args = parser.parse_args()

    if args.list_devices:
        list_audio_devices()
        sys.exit(0)
        
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("\nAsyncio run cancelled by user.")
    except Exception as e:
            print(f"\nUnhandled exception in asyncio run: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc() 