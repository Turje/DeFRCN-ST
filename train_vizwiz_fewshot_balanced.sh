#!/usr/bin/env bash

# VizWiz Few-Shot DeFRCN Training Script WITH CLASS BALANCING
# Addresses severe class imbalance (up to 451x ratio)
#
# Key difference from train_vizwiz_fewshot.sh:
# - Uses vizwiz_det_r101_base_balanced.yaml for base training
# - Adds RepeatFactorTrainingSampler to oversample rare classes
# - Lower learning rate for stability

set -e

# ============================== Configuration ============================== #
EXPERIMENT_NAME=${1:-"vizwiz_defrcn_balanced"}
DATA_ROOT="datasets/vizwiz"
OUTPUT_ROOT="outputs/vizwiz_fewshot_balanced"
WANDB_PROJECT="defrcn_vizwiz"

# Pretrained weights
IMAGENET_PRETRAIN="data/.pretrain_weights/ImageNetPretrained/MSRA/R-101.pkl"
IMAGENET_PRETRAIN_TORCH="data/.pretrain_weights/ImageNetPretrained/torchvision/resnet101-5d3b4d8f.pth"

# Training parameters
BASE_ITER=8000
NOVEL_ITER=2500
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
log_section "VizWiz Few-Shot DeFRCN Training WITH CLASS BALANCING"
echo "Experiment: ${EXPERIMENT_NAME}"
echo "Output: ${OUTPUT_ROOT}"
echo "W&B Project: ${WANDB_PROJECT}"
echo "Class Balancing: RepeatFactorTrainingSampler (ENABLED)"
echo ""

for FOLD in "${FOLDS[@]}"; do
    log_section "Processing Fold: ${FOLD}"
    
    FOLD_OUTPUT="${OUTPUT_ROOT}/${FOLD}"
    mkdir -p "${FOLD_OUTPUT}"
    
    # ======================== Phase 1: Base Training ======================= #
    log_section "Phase 1: Base Training (WITH Class Balancing)"
    
    BASE_OUTPUT="${FOLD_OUTPUT}/base_training"
    
    # Create base training config with class balancing
    BASE_CONFIG="${FOLD_OUTPUT}/config_base_${FOLD}_balanced.yaml"
    cat > "${BASE_CONFIG}" << EOF
_BASE_: "configs/vizwiz_det_r101_base.yaml"
MODEL:
  WEIGHTS: "${IMAGENET_PRETRAIN}"
  ROI_HEADS:
    NUM_CLASSES: 101
DATASETS:
  TRAIN: ('vizwiz_${FOLD}_train_base',)
  TEST: ('vizwiz_${FOLD}_val_base',)
  
# ========== CLASS BALANCING (Key Addition!) ========== #
DATALOADER:
  SAMPLER_TRAIN: "RepeatFactorTrainingSampler"
  REPEAT_THRESHOLD: 0.001  # Oversample classes with <0.1% frequency
  NUM_WORKERS: 4

SOLVER:
  IMS_PER_BATCH: 16
  BASE_LR: 0.01  # Reduced from 0.02 for stability
  WARMUP_ITERS: 500
  WARMUP_FACTOR: 0.001
  STEPS: (6000,)
  MAX_ITER: ${BASE_ITER}
  CHECKPOINT_PERIOD: 10000
  # Gradient clipping for stability with imbalanced data
  CLIP_GRADIENTS:
    ENABLED: True
    CLIP_TYPE: "norm"
    CLIP_VALUE: 1.0
    NORM_TYPE: 2.0

OUTPUT_DIR: "${BASE_OUTPUT}"
WANDB:
  ENABLED: True
  PROJECT: "${WANDB_PROJECT}"
  NAME: "${FOLD}_base_balanced"
EOF
    
    echo "Training base model with class balancing for ${FOLD}..."
    ${CONDA_PREFIX}/bin/python main.py --num-gpus ${NUM_GPUS} \
        --config-file "${BASE_CONFIG}" \
        --opts MODEL.WEIGHTS "${IMAGENET_PRETRAIN}" \
               OUTPUT_DIR "${BASE_OUTPUT}"
    
    # =================== Phase 2: Model Surgery ======================== #
    log_section "Phase 2: Model Surgery"
    
    echo "Performing model surgery: remove biased classification head..."
    ${CONDA_PREFIX}/bin/python tools/model_surgery.py \
        --dataset vizwiz \
        --method remove \
        --src-path "${BASE_OUTPUT}/model_final.pth" \
        --save-dir "${BASE_OUTPUT}"
    
    BASE_WEIGHT="${BASE_OUTPUT}/model_reset_remove.pth"
    
    # ==================== Phase 3: Few-Shot Fine-tuning ================= #
    log_section "Phase 3: Few-Shot Fine-tuning (Automatically Balanced!)"
    
    for SHOT in "${K_SHOTS[@]}"; do
        log_section "Training ${SHOT}-shot for ${FOLD}"
        
        NOVEL_OUTPUT="${FOLD_OUTPUT}/novel_${SHOT}shot"
        
        # Create novel training config
        # Note: Few-shot data is ALREADY balanced (k samples per class)
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
  NAME: "${FOLD}_novel_${SHOT}shot_balanced"
EOF
        
        echo "Training ${SHOT}-shot model (k=${SHOT} samples per class = balanced!)..."
        ${CONDA_PREFIX}/bin/python main.py --num-gpus ${NUM_GPUS} \
            --config-file "${NOVEL_CONFIG}" \
            --opts MODEL.WEIGHTS "${BASE_WEIGHT}" \
                   OUTPUT_DIR "${NOVEL_OUTPUT}" \
                   TEST.PCB_MODELPATH "${IMAGENET_PRETRAIN_TORCH}"
        
        echo "✅ ${SHOT}-shot training complete"
    done
    
    echo "✅ ${FOLD} complete!"
done

log_section "Training Complete!"
echo ""
echo "Class Balancing Strategy Applied:"
echo "  - Base training: RepeatFactorTrainingSampler (oversamples rare classes)"
echo "  - Few-shot: k-shot sampling (inherently balanced)"
echo ""
echo "Expected Improvement:"
echo "  - Base mAP: Higher (rare classes improved)"
echo "  - Novel mAP: Good (already balanced by k-shot)"
echo ""
echo "Check W&B dashboard: https://wandb.ai/your-username/${WANDB_PROJECT}"

