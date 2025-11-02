#!/usr/bin/env bash

# VizWiz Few-Shot DeFRCN FAST Training Script
# Optimized for quick experimentation (much shorter than paper)
#
# Configuration:
# - Base training: 2000 iterations (~2 epochs) - FAST!
# - Few-shot: 500 iterations (~5 epochs) - FAST!
# - W&B project: defrcn_gcp_cursor
# - Class balancing: ENABLED

set -e

# ============================== Configuration ============================== #
EXPERIMENT_NAME=${1:-"vizwiz_fast"}
DATA_ROOT="datasets/vizwiz"
OUTPUT_ROOT="outputs/vizwiz_fast"
WANDB_PROJECT="defrcn_gcp_cursor"  # Your requested W&B project name

# Pretrained weights
IMAGENET_PRETRAIN="data/.pretrain_weights/ImageNetPretrained/MSRA/R-101.pkl"
IMAGENET_PRETRAIN_TORCH="data/.pretrain_weights/ImageNetPretrained/torchvision/resnet101-5d3b4d8f.pth"

# FAST Training parameters (much shorter than paper!)
BASE_ITER=2000    # ~2 epochs (vs 8000 = 8 epochs in paper)
NOVEL_ITER=500    # ~5 epochs (vs 2500 = 25 epochs in paper)
FOLDS=("OD25_0")  # Just 1 fold for fast testing (add more if needed)
K_SHOTS=(10)      # Just 10-shot (remove 1-shot for speed)
NUM_GPUS=1

# ============================= Helper Functions ============================ #
log_section() {
    echo ""
    echo "======================================================================"
    echo "$1"
    echo "======================================================================"
    echo ""
}

# ================================ Main Loop ================================ #
log_section "VizWiz Few-Shot DeFRCN FAST Training"
echo "Experiment: ${EXPERIMENT_NAME}"
echo "Output: ${OUTPUT_ROOT}"
echo "W&B Project: ${WANDB_PROJECT}"
echo "Speed: FAST (2000 base iter, 500 novel iter)"
echo "Folds: ${FOLDS[@]}"
echo "K-shots: ${K_SHOTS[@]}"
echo ""
echo "⚡ This will be ~10x faster than paper!"
echo "   Base: 2000 iter (~1-2 hours)"
echo "   Novel: 500 iter (~20-30 min)"
echo "   Total per fold: ~2-3 hours"
echo ""

for FOLD in "${FOLDS[@]}"; do
    log_section "Processing Fold: ${FOLD}"
    
    FOLD_OUTPUT="${OUTPUT_ROOT}/${FOLD}"
    mkdir -p "${FOLD_OUTPUT}"
    
    # ======================== Phase 1: Base Training ======================= #
    log_section "Phase 1: FAST Base Training (2000 iter)"
    
    BASE_OUTPUT="${FOLD_OUTPUT}/base_training"
    
    # Create base training config with class balancing
    BASE_CONFIG="${FOLD_OUTPUT}/config_base_${FOLD}_fast.yaml"
    cat > "${BASE_CONFIG}" << EOF
_BASE_: "configs/vizwiz_det_r101_base.yaml"
MODEL:
  WEIGHTS: "${IMAGENET_PRETRAIN}"
  ROI_HEADS:
    NUM_CLASSES: 101
DATASETS:
  TRAIN: ('vizwiz_${FOLD}_train_base',)
  TEST: ('vizwiz_${FOLD}_val_base',)
  
# Class balancing
DATALOADER:
  SAMPLER_TRAIN: "RepeatFactorTrainingSampler"
  REPEAT_THRESHOLD: 0.001
  NUM_WORKERS: 2

SOLVER:
  IMS_PER_BATCH: 8  # Reduced from 16 for stability on single GPU
  BASE_LR: 0.005    # Lower LR for faster convergence
  WARMUP_ITERS: 200
  WARMUP_FACTOR: 0.001
  STEPS: (1500,)    # LR drop at 75% of training
  MAX_ITER: ${BASE_ITER}
  CHECKPOINT_PERIOD: 5000
  CLIP_GRADIENTS:
    ENABLED: True
    CLIP_TYPE: "norm"
    CLIP_VALUE: 1.0
    NORM_TYPE: 2.0

OUTPUT_DIR: "${BASE_OUTPUT}"
WANDB:
  ENABLED: True
  PROJECT: "${WANDB_PROJECT}"
  NAME: "${FOLD}_base_fast"
  CONFIG:
    experiment: "${EXPERIMENT_NAME}"
    base_iter: ${BASE_ITER}
    fold: "${FOLD}"
EOF
    
    echo "⚡ Training FAST base model (2000 iter) for ${FOLD}..."
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
    log_section "Phase 3: FAST Few-Shot Fine-tuning (500 iter)"
    
    for SHOT in "${K_SHOTS[@]}"; do
        log_section "⚡ Training FAST ${SHOT}-shot for ${FOLD}"
        
        NOVEL_OUTPUT="${FOLD_OUTPUT}/novel_${SHOT}shot"
        
        # Create novel training config
        NOVEL_CONFIG="${FOLD_OUTPUT}/config_novel_${SHOT}shot_${FOLD}_fast.yaml"
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
  IMS_PER_BATCH: 8
  BASE_LR: 0.01
  STEPS: (400,)
  MAX_ITER: ${NOVEL_ITER}
  CHECKPOINT_PERIOD: 5000
OUTPUT_DIR: "${NOVEL_OUTPUT}"
TEST:
  PCB_ENABLE: True
WANDB:
  ENABLED: True
  PROJECT: "${WANDB_PROJECT}"
  NAME: "${FOLD}_novel_${SHOT}shot_fast"
  CONFIG:
    experiment: "${EXPERIMENT_NAME}"
    novel_iter: ${NOVEL_ITER}
    k_shot: ${SHOT}
    fold: "${FOLD}"
EOF
        
        echo "⚡ Training FAST ${SHOT}-shot model (500 iter)..."
        ${CONDA_PREFIX}/bin/python main.py --num-gpus ${NUM_GPUS} \
            --config-file "${NOVEL_CONFIG}" \
            --opts MODEL.WEIGHTS "${BASE_WEIGHT}" \
                   OUTPUT_DIR "${NOVEL_OUTPUT}" \
                   TEST.PCB_MODELPATH "${IMAGENET_PRETRAIN_TORCH}"
        
        echo "✅ ${SHOT}-shot training complete"
    done
    
    echo "✅ ${FOLD} complete!"
done

log_section "🎉 FAST Training Complete!"
echo ""
echo "Results saved in: ${OUTPUT_ROOT}"
echo ""
echo "View on Weights & Biases:"
echo "  https://wandb.ai/your-username/${WANDB_PROJECT}"
echo ""
echo "⚡ Speed comparison:"
echo "  Paper: ~80 hours for 4 folds"
echo "  This run: ~2-3 hours for 1 fold"
echo "  Speedup: ~10x faster!"
echo ""
echo "Note: Results will be lower mAP than paper due to less training,"
echo "but sufficient for quick experimentation and debugging."

