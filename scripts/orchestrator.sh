#!/usr/bin/env bash
# ==============================================================================
# Benchmarking LLM Fine-Tuning Efficiency: Automated Experiment Orchestrator
# Compatible with: Ubuntu / Debian / Linux environments
# Supports: Single-GPU, Multi-GPU DDP, and Multi-GPU FSDP (for 7B/14B models)
#
# Usage:
#   bash scripts/orchestrator.sh [OPTIONS]
#
# Options:
#   --method <fft|lora|dora|qlora|all>   Target fine-tuning method(s) (supports e.g. dora/lora, dora,lora, default: all)
#   --model  <model_id|all>              Target model id, e.g. qwen2.5_0.5b (default: all)
#   --num-gpus <N>                       Number of GPUs to use per run (default: 1)
#   --use-fsdp                           Force FSDP mode (auto-enabled for 7B/14B FFT)
#   --profile                            Enable CUPTI/PyTorch profiling on runs
#   --skip-existing                      Skip runs that already have completed logs
#   --skip-eval                          Skip held-out and forgetting evaluation
#   --skip-svd                           Skip weight-space SVD rank analysis
#   --help                               Display this help message
# ==============================================================================

set -eo pipefail

# Visual Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Detect Python interpreter
PYTHON_BIN="${PYTHON:-python3}"
if ! command -v "$PYTHON_BIN" &> /dev/null; then
    PYTHON_BIN="python"
fi

# Detect system available GPUs via nvidia-smi
DETECTED_GPUS=1
if command -v nvidia-smi &> /dev/null; then
    COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader 2>/dev/null | head -n 1 || echo 1)
    if [[ "$COUNT" =~ ^[0-9]+$ ]]; then
        DETECTED_GPUS=$COUNT
    fi
fi

# Default parameters
TARGET_METHOD="all"
TARGET_MODEL="all"
NUM_GPUS=1
FORCE_FSDP=false
ENABLE_PROFILE=""
SKIP_EXISTING=false
SKIP_EVAL=false
SKIP_SVD=false

if [[ -f .env ]]; then
    source .env
fi
#export HF_HOME="${HF_HOME:-$(pwd)/data/huggingface_cache}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Parse command-line flags
while [[ $# -gt 0 ]]; do
    case "$1" in
        --method)
            if [[ "$TARGET_METHOD" == "all" ]]; then
                TARGET_METHOD="$2"
            else
                TARGET_METHOD="${TARGET_METHOD}/$2"
            fi
            shift 2
            ;;
        --model)
            TARGET_MODEL="$2"
            shift 2
            ;;
        --num-gpus)
            if [[ "$2" == "auto" || "$2" == "all" ]]; then
                NUM_GPUS=$DETECTED_GPUS
            else
                NUM_GPUS="$2"
            fi
            shift 2
            ;;
        --use-fsdp)
            FORCE_FSDP=true
            shift
            ;;
        --profile)
            ENABLE_PROFILE="--profile"
            shift
            ;;
        --skip-existing)
            SKIP_EXISTING=true
            shift
            ;;
        --skip-eval)
            SKIP_EVAL=true
            shift
            ;;
        --skip-svd)
            SKIP_SVD=true
            shift
            ;;
        --help)
            head -n 20 "$0" | tail -n 17
            exit 0
            ;;
        *)
            echo -e "${RED}[Error] Unknown argument: $1${NC}"
            exit 1
            ;;
    esac
done

# Ensure script is run from project root
if [[ ! -d "configs" || ! -d "src" ]]; then
    if [[ -d "../configs" && -d "../src" ]]; then
        cd ..
    else
        echo -e "${RED}[Error] Please run this script from the project root directory.${NC}"
        exit 1
    fi
fi

mkdir -p logs plots results checkpoints data/huggingface_cache

# Define methods to iterate
METHODS=()
if [[ "$TARGET_METHOD" == "all" ]]; then
    METHODS=("fft" "lora" "dora" "qlora")
else
    # Split by '/' or ',' or space into array
    read -r -a METHODS <<< "${TARGET_METHOD//[\/,]/ }"
fi

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE} LLM Fine-Tuning Efficiency Benchmark Orchestrator (Ubuntu)  ${NC}"
echo -e "${BLUE} Python:      $($PYTHON_BIN --version 2>&1)${NC}"
echo -e "${BLUE} GPUs Used:   ${NUM_GPUS} (Detected: ${DETECTED_GPUS})${NC}"
echo -e "${BLUE} Methods:     ${METHODS[*]}${NC}"
echo -e "${BLUE} Model:       ${TARGET_MODEL}${NC}"
echo -e "${BLUE} Profiler:    ${ENABLE_PROFILE:-Disabled}${NC}"
echo -e "${BLUE} Evaluation:  $([[ "$SKIP_EVAL" == true ]] && echo "Disabled" || echo "Enabled")${NC}"
echo -e "${BLUE} SVD Rank:    $([[ "$SKIP_SVD" == true ]] && echo "Disabled" || echo "Enabled")${NC}"
echo -e "${BLUE} Cache Dir:   $HF_HOME${NC}"
echo -e "${BLUE}============================================================${NC}\n"

START_TIME=$(date +%s)
TOTAL_RUNS=0
SUCCESSFUL_RUNS=0
FAILED_RUNS=0

# Iterate through selected methods and config files
for method in "${METHODS[@]}"; do
    CONFIG_DIR="configs/${method}"
    
    if [[ ! -d "$CONFIG_DIR" ]]; then
        echo -e "${YELLOW}[Warning] Directory $CONFIG_DIR not found, skipping.${NC}"
        continue
    fi

    for config_file in "$CONFIG_DIR"/*.yaml; do
        [[ -f "$config_file" ]] || continue
        
        config_name=$(basename "$config_file" .yaml)
        
        # Skip base.yaml template
        if [[ "$config_name" == "base" ]]; then
            continue
        fi

        # Filter by model if requested
        if [[ "$TARGET_MODEL" != "all" && "$config_name" != "$TARGET_MODEL" ]]; then
            continue
        fi

        TOTAL_RUNS=$((TOTAL_RUNS + 1))
        RUN_NAME="${config_name}_${method}"
        LOG_PATH="logs/${RUN_NAME}.jsonl"
        CHECKPOINT_DIR="checkpoints/${RUN_NAME}"

        # Extract base model name from yaml config
        BASE_MODEL=$($PYTHON_BIN -c "import yaml; print(yaml.safe_load(open('$config_file'))['model']['name'])" 2>/dev/null || echo "")
        TRAIN_SAMPLES=$($PYTHON_BIN -c "import yaml; print(yaml.safe_load(open('$config_file'))['data'].get('train_samples', 500))")
        VAL_SAMPLES=$($PYTHON_BIN -c "import yaml; print(yaml.safe_load(open('$config_file'))['data'].get('val_samples', 50))")
        MAX_LENGTH=$($PYTHON_BIN -c "import yaml; print(yaml.safe_load(open('$config_file'))['data']['max_length'])")

        # A nonempty log may belong to an interrupted run; require its final step and checkpoint.
        if [[ "$SKIP_EXISTING" == true && -s "$LOG_PATH" && -d "$CHECKPOINT_DIR" ]]; then
            if "$PYTHON_BIN" -c 'import json, sys, yaml; rows=open(sys.argv[1], encoding="utf-8").read().splitlines(); last=json.loads(rows[-1]); cfg=yaml.safe_load(open(sys.argv[2], encoding="utf-8")); sys.exit(0 if last["step"] >= cfg["training"]["max_steps"] and last.get("val_loss") is not None else 1)' "$LOG_PATH" "$config_file" 2>/dev/null; then
                echo -e "${YELLOW}[SKIP] Run $RUN_NAME already completed at $LOG_PATH.${NC}"
                SUCCESSFUL_RUNS=$((SUCCESSFUL_RUNS + 1))
                continue
            fi
        fi

        # Determine Multi-GPU Execution Mode (DDP vs. FSDP)
        RUN_DIST_MODE="Single-GPU"
        if [[ "$NUM_GPUS" -gt 1 ]]; then
            # Auto-enable FSDP for:
            # 1. FFT on 7B+ models (optimizer state is ~4x model size)
            # 2. 16-bit LoRA/DoRA on 13B/14B models (base model is ~29.5GB, saturating single 32GB GPUs in DDP)
            # 3. Explicitly requested with --use-fsdp
            IS_LARGE_FFT=$([[ "$method" == "fft" && "$config_name" =~ (7b|8b|13b|14b) ]] && echo true || echo false)
            IS_LARGE_16BIT_PEFT=$([[ ( "$method" == "lora" || "$method" == "dora" ) && "$config_name" =~ (13b|14b) ]] && echo true || echo false)

            if [[ "$FORCE_FSDP" == true || "$IS_LARGE_FFT" == true || "$IS_LARGE_16BIT_PEFT" == true ]]; then
                RUN_DIST_MODE="Multi-GPU (FSDP - $NUM_GPUS GPUs)"
                LAUNCH_CMD=("$PYTHON_BIN" "-m" "accelerate.commands.launch" "--use_fsdp" "--num_processes" "$NUM_GPUS" "--mixed_precision" "no" "--fsdp_use_orig_params" "True" "--fsdp_auto_wrap_policy" "TRANSFORMER_BASED_WRAP")
            else
                RUN_DIST_MODE="Multi-GPU (DDP - $NUM_GPUS GPUs)"
                LAUNCH_CMD=("$PYTHON_BIN" "-m" "accelerate.commands.launch" "--multi_gpu" "--num_processes" "$NUM_GPUS" "--mixed_precision" "bf16")
            fi
        else
            LAUNCH_CMD=("$PYTHON_BIN")
        fi

        echo -e "${GREEN}------------------------------------------------------------${NC}"
        echo -e "${GREEN}[RUN #${TOTAL_RUNS}] Launching: Method=${method^^} | Model=${config_name}${NC}"
        echo -e "${GREEN}Mode:       ${RUN_DIST_MODE}${NC}"
        echo -e "${GREEN}Config:     $config_file${NC}"
        echo -e "${GREEN}Checkpoint: $CHECKPOINT_DIR${NC}"
        echo -e "${GREEN}------------------------------------------------------------${NC}"

        RUN_START=$(date +%s)

        # 1. Execute training run
        if "${LAUNCH_CMD[@]}" src/train.py --config "$config_file" $ENABLE_PROFILE; then
            RUN_END=$(date +%s)
            DURATION=$((RUN_END - RUN_START))
            echo -e "${GREEN}[SUCCESS] Training finished for $RUN_NAME in ${DURATION}s.${NC}"
            SUCCESSFUL_RUNS=$((SUCCESSFUL_RUNS + 1))

            # 2. Run held-out evaluation & catastrophic forgetting (inference on single device)
            if [[ "$SKIP_EVAL" != true && -n "$BASE_MODEL" ]]; then
                echo -e "${BLUE}[Evaluation] Running held-out eval and forgetting on $RUN_NAME...${NC}"
                EVAL_JSON="results/${RUN_NAME}_eval.json"
                $PYTHON_BIN src/evaluate.py --model "$BASE_MODEL" --adapter "$CHECKPOINT_DIR" --train-samples "$TRAIN_SAMPLES" --val-samples "$VAL_SAMPLES" --max-length "$MAX_LENGTH" --output-json "$EVAL_JSON" || echo -e "${YELLOW}[Warning] Evaluation failed.${NC}"
            fi

            # 3. Run weight-space SVD and rank analysis (for PEFT methods)
            if [[ "$SKIP_SVD" != true && "$method" != "fft" && -n "$BASE_MODEL" ]]; then
                echo -e "${BLUE}[SVD Analysis] Running weight update SVD on $RUN_NAME...${NC}"
                SVD_JSON="results/${RUN_NAME}_svd.json"
                $PYTHON_BIN src/weight_analysis.py --model "$BASE_MODEL" --adapter "$CHECKPOINT_DIR" --output-json "$SVD_JSON" || echo -e "${YELLOW}[Warning] SVD analysis failed.${NC}"
            fi
            echo ""
        else
            echo -e "${RED}[FAILED] Experiment $RUN_NAME exited with an error.${NC}\n"
            FAILED_RUNS=$((FAILED_RUNS + 1))
        fi
    done
done

END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))

echo -e "\n${BLUE}============================================================${NC}"
echo -e "${BLUE} Benchmark Suite Finished in ${TOTAL_DURATION}s!${NC}"
echo -e "${BLUE} Total Runs:       ${TOTAL_RUNS}${NC}"
echo -e "${GREEN} Successful:       ${SUCCESSFUL_RUNS}${NC}"
if [[ $FAILED_RUNS -gt 0 ]]; then
    echo -e "${RED} Failed:           ${FAILED_RUNS}${NC}"
fi
echo -e "${BLUE}============================================================${NC}\n"

# Run Post-Training Analysis & Visualization
echo -e "${BLUE}[Post-Run Analysis] Generating Benchmark Table...${NC}"
$PYTHON_BIN -c "
import glob, sys
sys.path.insert(0, 'src')
from analysis import generate_summary_table
logs = sorted(glob.glob('logs/*.jsonl'))
if logs:
    generate_summary_table(logs)
" || true

echo -e "${BLUE}[Post-Run Analysis] Generating Comparative Plots...${NC}"
$PYTHON_BIN src/plot_curves.py || true

echo -e "\n${GREEN}All logs in 'logs/', checkpoints in 'checkpoints/', and results in 'results/'.${NC}"
