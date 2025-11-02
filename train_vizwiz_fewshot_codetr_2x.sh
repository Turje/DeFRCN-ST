#!/bin/bash
# VizWiz Few-Shot DeFRCN Training Script
# CO-DETR MATCH (2X) - Double the iterations for DeFRCN architecture
# 
# This script matches Co-DETR Lambda training but DOUBLED for DeFRCN:
# - Base: 3,472 iterations (~16 epochs) [Co-DETR: 1,736 / 8 epochs]
# - Novel: 1,400 iterations (~50 epochs for 10-shot) [Co-DETR: 700 / 25 epochs]
# - Time estimate: ~8-9 hours for 4 folds, k={1,10}

# Activate the correct conda environment
source ~/miniconda/etc/profile.d/conda.sh
conda activate defrcn-cu121

# Reduce CUDA fragmentation and avoid PCB OOM during in-training evaluation
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:256"

set -e  # Exit on error

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
ROOT_DIR="${HOME}/DeFRCN"
DATA_DIR="${ROOT_DIR}/datasets/vizwiz"
OUTPUT_BASE="${ROOT_DIR}/outputs_defrcn"
IMAGENET_PRETRAIN="${ROOT_DIR}/imagenet_pretrain/MSRA/R-101.pkl"
IMAGENET_PRETRAIN_TORCH="${ROOT_DIR}/imagenet_pretrain/torchvision/resnet101-5d3b4d8f.pth"

# Training parameters - CO-DETR MATCH (2X)
BASE_ITER=3472        # 2x Co-DETR's 1,736 iterations (16 epochs)
NOVEL_ITER=1400       # 2x Co-DETR's 700 iterations (50 epochs for 10-shot)

# Few-shot settings
FOLDS=("OD25_0" "OD25_1" "OD25_2" "OD25_3")  # All 4 folds
K_SHOTS=(1 3 5 10)    # All k-shots to match Co-DETR

# Hardware
NUM_GPUS=1

# Weights & Biases
export WANDB_PROJECT="gcp_DFRCN"
export WANDB_MODE="online"

# ============================================================================
# VERIFY SETUP
# ============================================================================

echo "========================================"
echo "VizWiz Few-Shot DeFRCN Training"
echo "CO-DETR MATCH (2X) Configuration"
echo "========================================"
echo ""
echo "Base iterations: ${BASE_ITER} (~16 epochs)"
echo "Novel iterations: ${NOVEL_ITER} (~50 epochs for 10-shot)"
echo "Folds: ${FOLDS[@]}"
echo "K-shots: ${K_SHOTS[@]}"
echo "Weights & Biases: ${WANDB_PROJECT}"
echo ""

# Check ImageNet pretrained weights
if [ ! -f "${IMAGENET_PRETRAIN}" ]; then
    echo "❌ ERROR: ImageNet pretrain not found at ${IMAGENET_PRETRAIN}"
    exit 1
fi

if [ ! -f "${IMAGENET_PRETRAIN_TORCH}" ]; then
    echo "❌ ERROR: ImageNet pretrain (torch) not found at ${IMAGENET_PRETRAIN_TORCH}"
    exit 1
fi

echo "✅ ImageNet pretrained weights found"
echo ""

# ============================================================================
# TRAINING PIPELINE
# ============================================================================

START_TIME=$(date +%s)

for FOLD in "${FOLDS[@]}"; do
    echo "========================================"
    echo "Processing Fold: ${FOLD}"
    echo "========================================"
    
    # ------------------------------------------------------------------------
    # STAGE 1: Base Training
    # ------------------------------------------------------------------------
    
    echo ""
    echo "📊 STAGE 1: Base Training (${BASE_ITER} iterations)"
    echo "----------------------------------------"
    
    BASE_CONFIG="${ROOT_DIR}/configs/vizwiz_det_r101_base_balanced.yaml"
    BASE_OUTPUT="${OUTPUT_BASE}/${FOLD}/base_model"
    
    mkdir -p "${BASE_OUTPUT}"
    
    # Check if already completed
    if [ -f "${BASE_OUTPUT}/model_final.pth" ]; then
        echo "✅ Base training already complete for ${FOLD} - SKIPPING"
    else
        # Set W&B run name
        export WANDB_NAME="${FOLD}_base"
        
        echo "Config: ${BASE_CONFIG}"
        echo "Output: ${BASE_OUTPUT}"
        echo "Iterations: ${BASE_ITER}"
        echo ""
        
        # Calculate STEPS values (60% and 80% of MAX_ITER)
        STEP1=$((BASE_ITER * 60 / 100))
        STEP2=$((BASE_ITER * 80 / 100))
        
        ${CONDA_PREFIX}/bin/python main.py --num-gpus ${NUM_GPUS} \
            --config-file "${BASE_CONFIG}" \
            --opts MODEL.WEIGHTS "${IMAGENET_PRETRAIN}" \
                   OUTPUT_DIR "${BASE_OUTPUT}" \
                   SOLVER.IMS_PER_BATCH 2 \
                   SOLVER.BASE_LR 0.0025 \
                   SOLVER.MAX_ITER ${BASE_ITER} \
                   SOLVER.STEPS "(${STEP1},${STEP2})" \
                   SOLVER.WARMUP_ITERS 1000 \
                   DATASETS.TRAIN "('vizwiz_${FOLD}_train_base',)" \
                   DATASETS.TEST "('vizwiz_${FOLD}_val_base',)"
        
        echo "✅ Base training complete for ${FOLD}"
    fi
    
    # ------------------------------------------------------------------------
    # Model Surgery: Remove classification head
    # ------------------------------------------------------------------------
    
    echo ""
    echo "🔧 Model Surgery: Removing classification head"
    echo "----------------------------------------"
    
    BASE_WEIGHT="${BASE_OUTPUT}/model_reset_remove.pth"
    
    # Check if already completed
    if [ -f "${BASE_WEIGHT}" ]; then
        echo "✅ Model surgery already complete for ${FOLD} - SKIPPING"
    else
        ${CONDA_PREFIX}/bin/python tools/model_surgery.py \
            --dataset vizwiz \
            --method remove \
            --src-path "${BASE_OUTPUT}/model_final.pth" \
            --save-dir "${BASE_OUTPUT}"
        
        if [ ! -f "${BASE_WEIGHT}" ]; then
            echo "❌ ERROR: Model surgery failed for ${FOLD}"
            exit 1
        fi
        
        echo "✅ Model surgery complete"
    fi
    
    # ------------------------------------------------------------------------
    # STAGE 2: Few-Shot Fine-Tuning
    # ------------------------------------------------------------------------
    
    for K in "${K_SHOTS[@]}"; do
        echo ""
                        ${CONDA_PREFIX}/bin/python main.py --num-gpus ${NUM_GPUS} \
                            --config-file "${NOVEL_CONFIG}" \
                            --opts MODEL.WEIGHTS "${BASE_WEIGHT}" \
                                   OUTPUT_DIR "${NOVEL_OUTPUT}" \
                                   SOLVER.IMS_PER_BATCH 1 \
                                   SOLVER.BASE_LR 0.001 \
                                   SOLVER.MAX_ITER ${NOVEL_ITER} \
                                   SOLVER.STEPS "(${NOVEL_STEP1},${NOVEL_STEP2})" \
                                   SOLVER.WEIGHT_DECAY 0.0001 \
                                   SOLVER.WEIGHT_DECAY_BIAS 0.0001 \
                                   SOLVER.CHECKPOINT_PERIOD 500 \
                                   TEST.EVAL_PERIOD 500 \
                                   TEST.PCB_ENABLE False \
                                   DATASETS.TRAIN "('vizwiz_${FOLD}_train_novel_${K}shot',)" \
                                   DATASETS.TEST "('vizwiz_${FOLD}_val_novel',)" \
                                   TEST.PCB_MODELPATH "${IMAGENET_PRETRAIN_TORCH}"
            echo "Config: ${NOVEL_CONFIG}"
            echo "Output: ${NOVEL_OUTPUT}"
            echo "K-shot: ${K}"
            echo "Iterations: ${NOVEL_ITER}"
            echo ""
            
            # Calculate STEPS values (60% and 80% of MAX_ITER)
            NOVEL_STEP1=$((NOVEL_ITER * 60 / 100))
            NOVEL_STEP2=$((NOVEL_ITER * 80 / 100))
            
            ${CONDA_PREFIX}/bin/python main.py --num-gpus ${NUM_GPUS} \
                --config-file "${NOVEL_CONFIG}" \
                --opts MODEL.WEIGHTS "${BASE_WEIGHT}" \
                       OUTPUT_DIR "${NOVEL_OUTPUT}" \
                       SOLVER.IMS_PER_BATCH 8 \
                       SOLVER.BASE_LR 0.001 \
                       SOLVER.MAX_ITER ${NOVEL_ITER} \
                       SOLVER.STEPS "(${NOVEL_STEP1},${NOVEL_STEP2})" \
                       SOLVER.WEIGHT_DECAY 0.0001 \
                       SOLVER.WEIGHT_DECAY_BIAS 0.0001 \
                       SOLVER.CHECKPOINT_PERIOD 500 \
                       TEST.EVAL_PERIOD 500 \
                       DATASETS.TRAIN "('vizwiz_${FOLD}_train_novel_${K}shot',)" \
                       DATASETS.TEST "('vizwiz_${FOLD}_val_novel',)" \
                       TEST.PCB_MODELPATH "${IMAGENET_PRETRAIN_TORCH}"
            
            echo "✅ Few-shot fine-tuning complete for ${FOLD} ${K}-shot"
        fi
    done
    
    echo ""
    echo "✅ Fold ${FOLD} complete!"
    echo ""
done

# ============================================================================
# SUMMARY
# ============================================================================

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))

echo ""
echo "========================================"
echo "🎉 TRAINING COMPLETE!"
echo "========================================"
echo ""
echo "Configuration:"
echo "  - Base iterations: ${BASE_ITER} (~16 epochs)"
echo "  - Novel iterations: ${NOVEL_ITER} (~50 epochs for 10-shot)"
echo "  - Folds trained: ${#FOLDS[@]}"
echo "  - K-shots: ${K_SHOTS[@]}"
echo "  - Total time: ${HOURS}h ${MINUTES}m"
echo ""
echo "Results saved in: ${OUTPUT_BASE}"
echo "W&B project: ${WANDB_PROJECT}"
echo ""
echo "Next steps:"
echo "  1. Check W&B dashboard for training curves"
echo "  2. Review metrics in ${OUTPUT_BASE}/*/metrics.json"
echo "  3. Run aggregate_results.py to compute average mAP"
echo ""
echo "To view results:"
echo "  python aggregate_results.py --output-dir ${OUTPUT_BASE}"
echo ""

