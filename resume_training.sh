#!/bin/bash
# Resume DeFRCN Training from where it stopped
# OD25_1: Base ✅ Done, Surgery ✅ Done, 1-shot 🔄 Resume

set -e

# Configuration
FOLD="OD25_1"
K_SHOT=1
NOVEL_ITER=1400
BASE_OUTPUT="./outputs_codetr_2x/${FOLD}/base_model"
NOVEL_OUTPUT="./outputs_codetr_2x/${FOLD}/novel_${K_SHOT}shot"
# Prefer the codetr_2x base weight, but fall back to an existing DeFRCN output
ALT_BASE_OUTPUT="./outputs_defrcn/${FOLD}/base_model"
if [ -f "${BASE_OUTPUT}/model_reset_remove.pth" ]; then
    BASE_WEIGHT="${BASE_OUTPUT}/model_reset_remove.pth"
elif [ -f "${ALT_BASE_OUTPUT}/model_final.pth" ]; then
    echo "Note: primary base weight not found; falling back to ${ALT_BASE_OUTPUT}/model_final.pth"
    BASE_WEIGHT="${ALT_BASE_OUTPUT}/model_final.pth"
else
    echo "Error: no base checkpoint found in ${BASE_OUTPUT} or ${ALT_BASE_OUTPUT}." >&2
    exit 1
fi
NOVEL_CONFIG="./configs/vizwiz_det_r101_novel_${K_SHOT}shot.yaml"
IMAGENET_PRETRAIN_TORCH="./imagenet_pretrain/torchvision/resnet101-5d3b4d8f.pth"

echo "========================================"
echo "RESUMING: OD25_1 1-shot training"
echo "========================================"
echo "From checkpoint: Last saved iteration"
echo "Target: ${NOVEL_ITER} iterations"
echo ""

# Force safer GPU-only runtime options to avoid CPU offload and OOM during eval
# Force safer GPU-only runtime options to avoid CPU offload and OOM during
# evaluation. Increase max_split_size_mb to reduce fragmentation and disable
# PCB during in-training evaluations (we can enable PCB for offline eval).
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:256"
# Runtime overrides applied to main.py invocations
RUNTIME_OPTS=(SOLVER.IMS_PER_BATCH 1 TEST.EVAL_PERIOD 10000 TEST.PCB_ENABLE False)

# Restart 1-shot training (checkpoint was deleted during cleanup)
${CONDA_PREFIX}/bin/python main.py --num-gpus 1 \
    --config-file "${NOVEL_CONFIG}" \
    --opts MODEL.WEIGHTS "${BASE_WEIGHT}" \
           OUTPUT_DIR "${NOVEL_OUTPUT}" \
           SOLVER.MAX_ITER ${NOVEL_ITER} \
           SOLVER.STEPS "(840,1120)" \
           DATASETS.TRAIN "('vizwiz_${FOLD}_train_novel_${K_SHOT}shot',)" \
           DATASETS.TEST "('vizwiz_${FOLD}_val_novel',)" \
           TEST.PCB_MODELPATH "${IMAGENET_PRETRAIN_TORCH}" "${RUNTIME_OPTS[@]}"

echo ""
echo "✅ OD25_1 1-shot training complete!"
echo ""

# Continue with 10-shot training
echo "========================================"
echo "Starting: OD25_1 10-shot training"
echo "========================================"

K_SHOT=10
NOVEL_OUTPUT="./outputs_codetr_2x/${FOLD}/novel_${K_SHOT}shot"
NOVEL_CONFIG="./configs/vizwiz_det_r101_novel_${K_SHOT}shot.yaml"

${CONDA_PREFIX}/bin/python main.py --num-gpus 1 \
    --config-file "${NOVEL_CONFIG}" \
    --opts MODEL.WEIGHTS "${BASE_WEIGHT}" \
           OUTPUT_DIR "${NOVEL_OUTPUT}" \
           SOLVER.MAX_ITER ${NOVEL_ITER} \
           SOLVER.STEPS "(840,1120)" \
           DATASETS.TRAIN "('vizwiz_${FOLD}_train_novel_${K_SHOT}shot',)" \
           DATASETS.TEST "('vizwiz_${FOLD}_val_novel',)" \
           TEST.PCB_MODELPATH "${IMAGENET_PRETRAIN_TORCH}"

echo ""
echo "✅ OD25_1 complete! Moving to OD25_2..."
echo ""

# Now run the full script for remaining folds (OD25_2, OD25_3)
echo "========================================"
echo "Continuing with OD25_2 and OD25_3"
echo "========================================"

./train_vizwiz_fewshot_codetr_2x.sh

