# ⚡ DeFRCN VizWiz FAST Training Guide

## Quick Start (2-3 hours instead of 80 hours!)

```bash
# 1. Activate environment
conda activate defrcn-cu121

# 2. Go to DeFRCN
cd ~/DeFRCN

# 3. Start FAST training
./train_vizwiz_fast.sh my_fast_experiment
```

## What's Different? (FAST vs Paper)

| Aspect | Paper (Full) | FAST Version | Speedup |
|--------|--------------|--------------|---------|
| Base iterations | 8,000 | **2,000** | 4x faster |
| Novel iterations | 2,500 | **500** | 5x faster |
| Folds | 4 | **1** (OD25_0) | 4x faster |
| K-shots | 1, 3, 5, 10 | **10 only** | 4x faster |
| Batch size | 16 | **8** | Better for 1 GPU |
| **Total time** | **~80 hours** | **~2-3 hours** | **~30x faster!** |

## Timeline Breakdown (Single GPU)

```
Phase 1: Base Training (2000 iter)
  ├─ Loading data: 2 min
  ├─ Training: 1-2 hours
  └─ Evaluation: 10 min
  Total: ~1-2 hours

Phase 2: Model Surgery
  └─ Remove classifier head: 1 min

Phase 3: Few-Shot Fine-tuning (500 iter)
  ├─ 10-shot training: 20-30 min
  └─ Evaluation: 5 min
  Total: ~30 min

Grand Total: ~2-3 hours for 1 fold ⚡
```

## Configuration Details

### W&B Settings
- **Project**: `defrcn_gcp_cursor` ✅ (as requested)
- **Run names**: `OD25_0_base_fast`, `OD25_0_novel_10shot_fast`
- **Logging**: Enabled with experiment metadata

### Training Settings
- **Base LR**: 0.005 (reduced for fast convergence)
- **Batch size**: 8 (optimized for single GPU)
- **Class balancing**: RepeatFactorTrainingSampler (ENABLED)
- **Gradient clipping**: Enabled for stability

### What to Expect

**Base mAP (after 2000 iter):**
- Common classes: 0.3-0.5
- Rare classes: 0.1-0.3
- Average: 0.2-0.4

**Novel mAP (10-shot, after 500 iter):**
- Novel classes: 0.2-0.4

**Note:** These are lower than paper results (~0.5-0.6) because we're training ~10x less, but sufficient for:
- ✅ Testing the pipeline
- ✅ Debugging issues
- ✅ Quick experimentation
- ✅ Verifying class balancing works

## If You Want More Accuracy

### Option 1: Train Longer (Medium Speed)
```bash
# Edit train_vizwiz_fast.sh:
BASE_ITER=5000    # ~5 epochs (was 2000)
NOVEL_ITER=1500   # ~15 epochs (was 500)

# Time: ~5-6 hours per fold
# mAP: Better, closer to paper
```

### Option 2: Train All Folds (Full Evaluation)
```bash
# Edit train_vizwiz_fast.sh:
FOLDS=("OD25_0" "OD25_1" "OD25_2" "OD25_3")

# Time: ~10-12 hours for 4 folds
# mAP: Averaged across folds (more robust)
```

### Option 3: Use Full Training
```bash
# Use the balanced script (recommended for final results)
./train_vizwiz_fewshot_balanced.sh final_experiment

# Time: ~60-80 hours for 4 folds
# mAP: Paper-level results
```

## Monitoring Training

### In Terminal (Real-time)
```bash
# If running in background
tail -f training.log

# Check GPU usage
watch -n 1 nvidia-smi
```

### Weights & Biases Dashboard
```bash
# Open in browser:
https://wandb.ai/your-username/defrcn_gcp_cursor

# You'll see:
# - Loss curves (should decrease)
# - mAP metrics (updated each eval)
# - Training speed
# - GPU utilization
```

## Troubleshooting

### "CUDA out of memory"
```bash
# Reduce batch size in train_vizwiz_fast.sh:
IMS_PER_BATCH: 4  # Changed from 8
```

### "Low mAP even after training"
This is **expected** with fast training! Solutions:
1. ✅ Check if loss is decreasing (main indicator)
2. ✅ Verify class balancing is working (check W&B)
3. ✅ Increase iterations if you want higher mAP
4. ✅ Use full training script for final results

### "Training is slow"
On single GPU with batch size 8:
- Base: ~0.5-1 sec/iter → 2000 iter = ~1-2 hours ✅
- Novel: ~0.3-0.5 sec/iter → 500 iter = ~20-30 min ✅

This is normal!

## Files Created

```
outputs/vizwiz_fast/
└── OD25_0/
    ├── base_training/
    │   ├── model_final.pth
    │   ├── model_reset_remove.pth
    │   ├── metrics.json
    │   └── log.txt
    └── novel_10shot/
        ├── model_final.pth
        ├── metrics.json
        └── log.txt
```

## Next Steps After Fast Training

1. **Check W&B dashboard** - Verify training is working
2. **Review mAP** - Should be >0.2 even with fast training
3. **If results look good** - Run full training for paper
4. **If issues found** - Debug with fast training (saves time!)

## Commands Summary

```bash
# Fast training (2-3 hours)
./train_vizwiz_fast.sh my_experiment

# Background training
nohup ./train_vizwiz_fast.sh my_experiment > training.log 2>&1 &

# Monitor
tail -f training.log

# Check W&B
# https://wandb.ai/your-username/defrcn_gcp_cursor
```

---

**Ready to start! This will be ~30x faster than the paper.** ⚡

**W&B Project**: `defrcn_gcp_cursor` ✅
**Time**: ~2-3 hours for 1 fold ✅
**Purpose**: Quick experimentation and debugging ✅

