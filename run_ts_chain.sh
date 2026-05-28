#!/bin/bash
# Sequential template search for multiple generation models.
# Run: bash run_ts_chain.sh
# Qwen2.5-7B is already running separately; this chains Llama → Mistral → Gemma(optional).

set -e
PYTHON=/home/parvk/miniconda3/envs/newt/bin/python
BASE=/home/parvk/smc-llms

run_model() {
    local model="$1"
    local log="$2"
    echo "=== Starting $model ===" | tee -a "$log"
    $PYTHON "$BASE/template_search.py" \
        --model "$model" \
        --semantic \
        --smc-samples 200 \
        --batch-size 8 \
        --cls-device cuda:1 \
        >> "$log" 2>&1
    echo "=== Done $model ===" | tee -a "$log"
}

# Wait for Qwen2.5-7B run (PID may still be running)
echo "Waiting for Qwen2.5-7B (PID 234864) to finish..."
while kill -0 234864 2>/dev/null; do sleep 30; done
echo "Qwen2.5-7B done."

run_model "meta-llama/Llama-3.1-8B-Instruct"    "$BASE/results/ts_llama8b.log"
run_model "mistralai/Mistral-7B-Instruct-v0.2"   "$BASE/results/ts_mistral7b.log"

# Gemma: only runs if the model is accessible
if $PYTHON -c "from huggingface_hub import snapshot_download; snapshot_download('google/gemma-2-9b-it', ignore_patterns=['*.bin'])" 2>/dev/null; then
    run_model "google/gemma-2-9b-it" "$BASE/results/ts_gemma9b.log"
else
    echo "Gemma-2-9b-it not accessible (gated). Skipping." | tee -a "$BASE/results/ts_gemma9b.log"
fi

echo "All done."
