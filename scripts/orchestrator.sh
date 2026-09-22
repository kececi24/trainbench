#!/usr/bin/env bash
# ==============================================================================
# Benchmarking LLM Fine-Tuning Efficiency: Automated Experiment Orchestrator
# Compatible with: Ubuntu / Debian / Linux environments
#
# Usage:
#   bash scripts/orchestrator.sh [OPTIONS]
#
# Options:
#   --method <fft|lora|dora|qlora|all>   Target fine-tuning method (default: all)
#   --model  <model_id|all>              Target model id, e.g. qwen2.5_0.5b (default: all)
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

# Default parameters
TARGET_METHOD="all"
TARGET_MODEL="all"
ENABLE_PROFILE=""
SKIP_EXISTING=false
SKIP_EVAL=false
SKIP_SVD=false

export HF_HOME="$(pwd)/data/huggingface_cache"

# Parse command-line flags
while [[ $# -gt 0 ]]; do
    case "$1" in
        --method)
            TARGET_METHOD="$2"
            shift 2
            ;;
        --model)
            TARGET_MODEL="$2"
            shift 2
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
            head -n 18 "$0" | tail -n 15
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
    METHODS=("$TARGET_METHOD")
fi

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE} LLM Fine-Tuning Efficiency Benchmark Orchestrator (Ubuntu)  ${NC}"
echo -e "${BLUE} Python:      $($PYTHON_BIN --version 2>&1)${NC}"
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

        # Check if already completed
        if [[ "$SKIP_EXISTING" == true && -s "$LOG_PATH" ]]; then
            echo -e "${YELLOW}[SKIP] Run $RUN_NAME already completed at $LOG_PATH.${NC}"
            SUCCESSFUL_RUNS=$((SUCCESSFUL_RUNS + 1))
            continue
        fi

        echo -e "${GREEN}------------------------------------------------------------${NC}"
        echo -e "${GREEN}[RUN #${TOTAL_RUNS}] Launching: Method=${method^^} | Model=${config_name}${NC}"
        echo -e "${GREEN}Config:     $config_file${NC}"
        echo -e "${GREEN}Checkpoint: $CHECKPOINT_DIR${NC}"
        echo -e "${GREEN}------------------------------------------------------------${NC}"

        RUN_START=$(date +%s)

        # 1. Execute training run (saves logs & checkpoints)
        if $PYTHON_BIN src/train.py --config "$config_file" $ENABLE_PROFILE; then
            RUN_END=$(date +%s)
            DURATION=$((RUN_END - RUN_START))
            echo -e "${GREEN}[SUCCESS] Training finished for $RUN_NAME in ${DURATION}s.${NC}"
            SUCCESSFUL_RUNS=$((SUCCESSFUL_RUNS + 1))

            # 2. Run held-out evaluation & catastrophic forgetting
            if [[ "$SKIP_EVAL" != true && -n "$BASE_MODEL" ]]; then
                echo -e "${BLUE}[Evaluation] Running held-out eval and forgetting on $RUN_NAME...${NC}"
                EVAL_JSON="results/${RUN_NAME}_eval.json"
                $PYTHON_BIN src/evaluate.py --model "$BASE_MODEL" --adapter "$CHECKPOINT_DIR" --output-json "$EVAL_JSON" || echo -e "${YELLOW}[Warning] Evaluation failed.${NC}"
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
