# DeFRCN Training Setup - Complete Guide

## 🎯 Overview

This setup mirrors the Co-DETR configuration on Lambda AI:
- **4 folds**: OD25_0, OD25_1, OD25_2, OD25_3
- **5 stages per fold**: Base + 1/3/5/10-shot
- **W&B logging**: Project `DeFRCN_GCP_New`
- **Estimated time**: ~8-10 hours total

---

## ✅ Pre-Training Verification

Run comprehensive verification before starting:

```bash
cd ~/DeFRCN
python3 comprehensive_verification.py
```

**Verification checks:**
- ✅ GPU availability (1x NVIDIA L4)
- ✅ Data structure (all 4 folds present)
- ✅ Annotation files (75 base + 25 novel classes)
- ✅ Config files (NUM_CLASSES correct)
- ✅ Training scripts
- ✅ Environment (PyTorch, Detectron2, W&B)

---

## 🚀 Quick Start (Easiest)

**One command to start everything:**

```bash
cd ~/DeFRCN
bash start_defrcn_training.sh
```

This will:
1. Stop any old training processes
2. Optionally clean old outputs
3. Start W&B auto-logger
4. Start training (all 4 folds)
5. Show status and monitoring options

---

## 📊 Monitoring Training

### Option 1: Comprehensive Live Monitor (Recommended)

```bash
cd ~/DeFRCN
python3 monitor_defrcn_comprehensive.py
```

Shows:
- Training process status (PID, CPU, memory)
- W&B logger status
- Current fold/stage
- Latest training iterations
- Fold completion status
- W&B dashboard link

Updates every 15 seconds. Press `Ctrl+C` to stop monitoring (training continues).

### Option 2: Log File

```bash
tail -f ~/DeFRCN/training_DeFRCN_GCP_New.log
```

### Option 3: W&B Dashboard

🔗 https://wandb.ai/sturjem000-the-city-university-of-new-york/DeFRCN_GCP_New

---

## 📋 Training Pipeline Details

### Configuration

```bash
# Base Training
- Iterations: 3,472 (~16 epochs)
- Batch size: 2
- Learning rate: 0.0025
- Classes: 75 (base only)

# Few-Shot Training (1, 3, 5, 10-shot)
- Iterations: 1,400 (~50 epochs for 10-shot)
- Batch size: 8
- Learning rate: 0.001
- Classes: 100 (base + novel)
```

### W&B Run Names

Each fold and stage gets its own W&B run:
- `OD25_0_base`
- `OD25_0_novel_1shot`
- `OD25_0_novel_3shot`
- `OD25_0_novel_5shot`
- `OD25_0_novel_10shot`
- (repeated for OD25_1, OD25_2, OD25_3)

**Total: 20 W&B runs** (4 folds × 5 stages)

---

## 🔧 Manual Control

### Start W&B Logger Only

```bash
cd ~/DeFRCN
nohup python3 wandb_auto_logger_defrcn.py > wandb_logger_defrcn.log 2>&1 &
```

### Start Training Only

```bash
cd ~/DeFRCN
nohup bash train_vizwiz_fewshot_codetr_2x.sh > training_DeFRCN_GCP_New.log 2>&1 &
```

### Stop Everything

```bash
pkill -f 'python.*main.py'
pkill -f 'wandb_auto_logger_defrcn'
```

### Check Status

```bash
# Check training
ps aux | grep 'python.*main.py' | grep -v grep

# Check W&B logger
ps aux | grep 'wandb_auto_logger_defrcn' | grep -v grep
```

---

## 📁 Output Structure

```
outputs_codetr_2x/
├── OD25_0/
│   ├── base_model/
│   │   ├── model_final.pth
│   │   ├── model_reset_remove.pth
│   │   ├── metrics.json
│   │   └── log.txt
│   ├── novel_1shot/
│   ├── novel_3shot/
│   ├── novel_5shot/
│   └── novel_10shot/
├── OD25_1/
├── OD25_2/
└── OD25_3/
```

---

## ⏱️  Time Estimates

| Stage | Time per Fold | Total (4 Folds) |
|-------|--------------|-----------------|
| Base | ~1.5-2h | ~6-8h |
| 1-shot | ~15-20min | ~1h |
| 3-shot | ~15-20min | ~1h |
| 5-shot | ~15-20min | ~1h |
| 10-shot | ~15-20min | ~1h |
| **Total** | **~2-2.5h** | **~8-10h** |

---

## 🎯 Expected Performance

Based on corrected configurations:

### Base Training (75 classes)
- **Expected mAP**: 15-25%
- **Expected mAP@50**: 25-35%

### Few-Shot Training (100 classes)
- **1-shot**: 8-12% mAP
- **3-shot**: 10-14% mAP
- **5-shot**: 12-16% mAP
- **10-shot**: 14-18% mAP

*(Much better than previous 0.05-7% due to NUM_CLASSES fix)*

---

## 🔍 Troubleshooting

### Training not starting?

```bash
# Check log for errors
tail -50 ~/DeFRCN/training_DeFRCN_GCP_New.log

# Re-run verification
python3 comprehensive_verification.py
```

### W&B not logging?

```bash
# Check logger status
tail -50 ~/DeFRCN/wandb_logger_defrcn.log

# Restart logger
pkill -f 'wandb_auto_logger_defrcn'
nohup python3 wandb_auto_logger_defrcn.py > wandb_logger_defrcn.log 2>&1 &
```

### Training stopped unexpectedly?

```bash
# Check if process is still running
ps aux | grep 'python.*main.py'

# Check last error in log
tail -100 ~/DeFRCN/training_DeFRCN_GCP_New.log
```

---

## 📊 Comparison with Co-DETR

| Aspect | DeFRCN (GCP) | Co-DETR (Lambda AI) |
|--------|--------------|---------------------|
| GPU | 1x L4 (23.6 GB) | 8x A100 |
| Backbone | ResNet101 | Swin-B |
| Time | ~8-10h | ~16-20h |
| W&B Project | DeFRCN_GCP_New | CODETR_LAMBDA_New |
| Folds | 4 | 4 |
| K-shots | 1,3,5,10 | 1,3,5,10 |

Both setups:
- ✅ Correct NUM_CLASSES (75 base, 100 total)
- ✅ Auto-switching W&B logging
- ✅ Comprehensive monitoring
- ✅ All 4 folds × 5 stages

---

## ✅ Final Checklist

Before starting training, confirm:

- [x] Verification passed (`comprehensive_verification.py`)
- [x] W&B project: `DeFRCN_GCP_New`
- [x] Training script: all 4 folds, k-shots 1,3,5,10
- [x] Scripts executable (`chmod +x *.sh *.py`)
- [x] Ready to run for ~8-10 hours

**You're all set! Run:**
```bash
bash start_defrcn_training.sh
```

---

## 📞 Quick Reference Commands

```bash
# Start everything
bash start_defrcn_training.sh

# Monitor live
python3 monitor_defrcn_comprehensive.py

# Check logs
tail -f training_DeFRCN_GCP_New.log

# W&B dashboard
https://wandb.ai/sturjem000-the-city-university-of-new-york/DeFRCN_GCP_New

# Stop training
pkill -f 'python.*main.py'
```

---

**Last updated**: 2025-10-26
**Status**: ✅ Ready to train

