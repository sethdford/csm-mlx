#!/bin/bash

# Activate the virtual environment if necessary
# source ../mlx_bogle_env/bin/activate

# Path to the streaming script (relative to this script's location)
SCRIPT_PATH="run_streaming_csm_mlx.py"

# --- Configuration ---
# STT (Faster Whisper) Parameters
STT_MODEL_SIZE="large-v3" # tiny.en, base, small, medium, large-v1, large-v2, large-v3
STT_DEVICE="mps"       # cpu, cuda, mps
STT_COMPUTE_TYPE="float16" # default, auto, float16, float32, int8, int8_float16
STT_LANG="en"          # Language code (e.g., en, de, fr, es) or 'auto' for detection
ONLINE_MIN_CHUNK=0.2   # Minimum audio chunk size in seconds for online processing

# TTS (CSM MLX) Parameters
TTS_MODEL_REPO="senstella/csm-1b-mlx" # HF repo for the CSM model
# TTS_QUANTIZE="--quantize"          # Uncomment to enable quantization (uses default 4-bit, 64 group size)
TTS_QUANTIZE=""
# ADAPTER_FILE="../finetunning/bogle_finetuned_mlx_adapters/adapters.safetensors" # Path to your adapter file
# ADAPTER_ARG="--adapter-file $ADAPTER_FILE" # Uncomment if using an adapter
ADAPTER_ARG=""
TTS_SPEAKER=0          # Speaker ID for TTS
TTS_TEMP=0.7           # Temperature for TTS sampling

# LLM (MLX LM) Parameters
# LLM_MODEL_PATH="mlx-community/Mistral-7B-Instruct-v0.2-4bit" # Quantized Mistral
# LLM_MODEL_PATH="mlx-community/Mistral-7B-Instruct-v0.2"      # Unquantized Mistral
LLM_MODEL_PATH="mlx-community/Phi-3-mini-4k-instruct-4bit" # Quantized Phi-3 Mini
# LLM_MODEL_PATH="mlx-community/phi-3-mini-128k-instruct-4bit" # Quantized Phi-3 Mini 128k
# LLM_MODEL_PATH="mlx-community/Llama-3-8B-Instruct-4bit"   # Quantized Llama-3
LLM_MAX_TOKENS=200      # Max tokens for LLM generation
LLM_TEMP=0.7            # Temperature for LLM sampling

# Audio Device Parameters (Optional - find IDs using --list-devices)
# INPUT_DEVICE=your_input_id
# OUTPUT_DEVICE=your_output_id
# INPUT_DEVICE_ARG="--input-device $INPUT_DEVICE"
# OUTPUT_DEVICE_ARG="--output-device $OUTPUT_DEVICE"
INPUT_DEVICE_ARG=""
OUTPUT_DEVICE_ARG=""

# Output WAV file (Optional)
# OUTPUT_WAV="conversation_output.wav"
# OUTPUT_WAV_ARG="--output-file $OUTPUT_WAV"
OUTPUT_WAV_ARG=""

# --- Build Command ---
COMMAND=(
    "python" "$SCRIPT_PATH"
    "--stt-model-size" "$STT_MODEL_SIZE"
    "--stt-device" "$STT_DEVICE"
    "--stt-compute-type" "$STT_COMPUTE_TYPE"
    "--stt-lang" "$STT_LANG"
    "--online-min-chunk-seconds" "$ONLINE_MIN_CHUNK"
    "--model-repo" "$TTS_MODEL_REPO"
    $TTS_QUANTIZE # Add quantization flag if set
    $ADAPTER_ARG # Add adapter arg if set
    "--speaker" "$TTS_SPEAKER"
    "--temperature" "$TTS_TEMP"
    "--llm-model-path" "$LLM_MODEL_PATH"
    "--llm-max-tokens" "$LLM_MAX_TOKENS"
    "--llm-temp" "$LLM_TEMP"
    $INPUT_DEVICE_ARG
    $OUTPUT_DEVICE_ARG
    $OUTPUT_WAV_ARG
)

# --- Execute Command ---
echo "Running command:"
# Print command array elements one per line for readability
printf '%s\n' "${COMMAND[@]}"
echo "---------------------"

"${COMMAND[@]}"

# Deactivate environment if you activated it
# deactivate 