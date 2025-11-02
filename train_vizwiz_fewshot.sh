#!/usr/bin/env bash

# VizWiz Few-Shot DeFRCN Training Script
# Following OD-25ᵢ protocol with optimized configuration
# 
# Configuration:
# - Base training: 8 epochs (optimized from 12)
# - Few-shot training: 25 epochs (optimized from 50) 
# - K-shots: 1, 10 (most informative comparison)
# - All 4 folds: OD25_0, OD25_1, OD25_2, OD25_3
# - Logs to Weights & Biases project: defrcn_vizwiz

set -e  # Exit on error

# ============================== Configuration ============================== #
EXPERIMENT_NAME=${1:-"vizwiz_defrcn_fewshot"}
DATA_ROOT="datasets/vizwiz"
OUTPUT_ROOT="outputs/vizwiz_fewshot"
WANDB_PROJECT="defrcn_vizwiz"

# Pretrained weights
IMAGENET_PRETRAIN="data/.pretrain_weights/ImageNetPretrained/MSRA/R-101.pkl"
IMAGENET_PRETRAIN_TORCH="data/.pretrain_weights/ImageNetPretrained/torchvision/resnet101-5d3b4d8f.pth"

# Training parameters (optimized for speed while maintaining quality)
BASE_ITER=8000    # ~8 epochs with batch size 16
NOVEL_ITER=2500   # ~25 epochs  
FOLDS=("OD25_0" "OD25_1" "OD25_2" "OD25_3")
K_SHOTS=(1 10)
NUM_GPUS=1  # Changed from 8 to 1 for GCP single-GPU VM

# ============================= Helper Functions ============================ #
log_section() {
    echo ""
    echo "======================================================================"
    echo "$1"
    echo "======================================================================"
    echo ""
}

# ================================ Main Loop ================================ #
log_section "Starting VizWiz Few-Shot DeFRCN Training"
echo "Experiment: ${EXPERIMENT_NAME}"
echo "Output: ${OUTPUT_ROOT}"
echo "W&B Project: ${WANDB_PROJECT}"
echo "Folds: ${FOLDS[@]}"
echo "K-shots: ${K_SHOTS[@]}"
echo ""

for FOLD in "${FOLDS[@]}"; do
    log_section "Processing Fold: ${FOLD}"
    
    FOLD_OUTPUT="${OUTPUT_ROOT}/${FOLD}"
    mkdir -p "${FOLD_OUTPUT}"
    
    # ======================== Phase 1: Base Training ======================= #
    log_section "Phase 1: Base Training on 75 base categories"
    
    BASE_OUTPUT="${FOLD_OUTPUT}/base_training"
    
    # Create base training config dynamically
    BASE_CONFIG="${FOLD_OUTPUT}/config_base_${FOLD}.yaml"
    cat > "${BASE_CONFIG}" << EOF
_BASE_: "configs/vizwiz_det_r101_base.yaml"
MODEL:
  WEIGHTS: "${IMAGENET_PRETRAIN}"
  ROI_HEADS:
    NUM_CLASSES: 101
DATASETS:
  TRAIN: ('vizwiz_${FOLD}_train_base',)
  TEST: ('vizwiz_${FOLD}_val_base',)
SOLVER:
  IMS_PER_BATCH: 16
  BASE_LR: 0.02
  STEPS: (6000,)
  MAX_ITER: ${BASE_ITER}
  CHECKPOINT_PERIOD: 10000
OUTPUT_DIR: "${BASE_OUTPUT}"
WANDB:
  ENABLED: True
  PROJECT: "${WANDB_PROJECT}"
  NAME: "${FOLD}_base_training"
EOF
    
    echo "Training base model for ${FOLD}..."
    python main.py --num-gpus ${NUM_GPUS} \
        --config-file "${BASE_CONFIG}" \
        --opts MODEL.WEIGHTS "${IMAGENET_PRETRAIN}" \
               OUTPUT_DIR "${BASE_OUTPUT}"
    
    # =================== Phase 2: Model Surgery ======================== #
    log_section "Phase 2: Model Surgery (preparing for few-shot)"
    
    echo "Performing model surgery: remove classification head..."
    python tools/model_surgery.py \
        --dataset vizwiz \
        --method remove \
        --src-path "${BASE_OUTPUT}/model_final.pth" \
        --save-dir "${BASE_OUTPUT}"
    
    BASE_WEIGHT="${BASE_OUTPUT}/model_reset_remove.pth"
    
    # ==================== Phase 3: Few-Shot Fine-tuning ================= #
    log_section "Phase 3: Few-Shot Fine-tuning on 25 novel categories"
    
    for SHOT in "${K_SHOTS[@]}"; do
        log_section "Training ${SHOT}-shot for ${FOLD}"
        
        NOVEL_OUTPUT="${FOLD_OUTPUT}/novel_${SHOT}shot"
        
        # Create novel training config dynamically
        NOVEL_CONFIG="${FOLD_OUTPUT}/config_novel_${SHOT}shot_${FOLD}.yaml"
        cat > "${NOVEL_CONFIG}" << EOF
_BASE_: "configs/vizwiz_det_r101_base.yaml"
MODEL:
  WEIGHTS: "${BASE_WEIGHT}"
  ROI_HEADS:
    NUM_CLASSES: 101
DATASETS:
  TRAIN: ('vizwiz_${FOLD}_train_novel_${SHOT}shot',)
  TEST: ('vizwiz_${FOLD}_val_novel_${SHOT}shot',)
SOLVER:
  IMS_PER_BATCH: 16
  BASE_LR: 0.02
  STEPS: (2000,)
  MAX_ITER: ${NOVEL_ITER}
  CHECKPOINT_PERIOD: 10000
OUTPUT_DIR: "${NOVEL_OUTPUT}"
TEST:
  PCB_ENABLE: True
WANDB:
  ENABLED: True
  PROJECT: "${WANDB_PROJECT}"
  NAME: "${FOLD}_novel_${SHOT}shot"
EOF
        
        echo "Training ${SHOT}-shot model for ${FOLD}..."
        python main.py --num-gpus ${NUM_GPUS} \
            --config-file "${NOVEL_CONFIG}" \
            --opts MODEL.WEIGHTS "${BASE_WEIGHT}" \
                   OUTPUT_DIR "${NOVEL_OUTPUT}" \
                   TEST.PCB_MODELPATH "${IMAGENET_PRETRAIN_TORCH}"
        
        # Clean up to save space (optional)
        echo "Cleaning up intermediate checkpoints..."
        find "${NOVEL_OUTPUT}" -name "model_*.pth" ! -name "model_final.pth" -delete 2>/dev/null || true
        
        echo "✅ ${SHOT}-shot training complete for ${FOLD}"
    done
    
    echo "✅ ${FOLD} complete!"
    echo ""
done

# ============================== Summarize Results ============================== #
log_section "Training Complete for All Folds!"

echo "Results saved in: ${OUTPUT_ROOT}"
echo ""
echo "To view results in Weights & Biases:"
echo "  https://wandb.ai/your-username/${WANDB_PROJECT}"
echo ""
echo "To extract and summarize all results, run:"
echo "  python tools/extract_vizwiz_results.py --res-dir ${OUTPUT_ROOT}"
echo ""
log_section "Done!"

