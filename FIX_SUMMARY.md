# 🚨 CRITICAL BUG FIX: Low mAP Root Cause

## Problem Identified

Your training had **catastrophically low mAP** due to a class mismatch bug:

###  The Bug:
1. **Dataset registration** claimed: **100 classes for ALL splits** (base AND novel)
2. **Actual data:**
   - BASE split: **75 classes** only
   - NOVEL split: **25 classes** only  
3. **Model config:** `NUM_CLASSES: 101` (to handle IDs 0-100)

This caused **wrong class mappings** where:
- Model learned incorrect class IDs
- Predictions mapped to wrong classes  
- mAP tanked (0.05% for 10-shot novel!)

### Evidence:
```
Category ids in annotations are not in [1, #categories]! We'll apply a mapping for you.
```

## The Fix

✅ **Fixed dataset registration** (`defrcn/data/datasets/vizwiz.py`):
- Now dynamically loads ACTUAL categories from annotation files
- Registers each split with ONLY its real classes
- Base: 75 classes, Novel: 25 classes

## What Needs to Happen Now

**CRITICAL**: You must **RESTART training from scratch** with the fixed registration!

The current training (iteration 2879/3472) is using the BROKEN registration and will continue to have low mAP even if it completes.

###  Stop Current Training:
```bash
# Kill the running training process
kill $(ps aux | grep "python.*main.py" | grep -v grep | awk '{print $2}')
```

### Clean Old Outputs:
```bash
cd ~/DeFRCN
rm -rf outputs_codetr_2x/OD25_1/*  # Clean broken training
```

### Update Configs:
You also need to fix `NUM_CLASSES` in configs:

**For base training** (`configs/vizwiz_det_r101_base_balanced.yaml`):
```yaml
ROI_HEADS:
  NUM_CLASSES: 75  # Was 101, should be 75 for base classes
```

**For novel training** (`configs/vizwiz_det_r101_novel_1shot.yaml` and similar):
```yaml
ROI_HEADS:
  NUM_CLASSES: 100  # Should be 100 (75 base + 25 novel)
```

### Restart Training:
```bash
cd ~/DeFRCN
./train_vizwiz_fewshot_codetr_2x.sh 2>&1 | tee training_codetr_2x_FIXED_v2.log &
```

## Expected Results After Fix

- **Base training mAP**: 20-35% (was 7.7%)
- **1-shot novel mAP**: 10-15% (was 1.8%)
- **10-shot novel mAP**: 15-25% (was 0.05%!)

## Co-DETR on Lambda AI

The SAME bug likely exists in your Co-DETR training! You need to check:
1. How many classes are in the data vs config
2. If there's a similar class mismatch

Check Co-DETR's dataset registration and config files for the same issue.

## W&B Naming Issue

The W&B naming follows the fold correctly (`OD25_1_base_codetr2x`), but you mentioned it's not matching the run. Please clarify:
- What name do you expect?
- What name is showing in W&B?

---

**Action Required**: Please confirm if you want to:
1. Stop current training
2. Apply the fix
3. Restart from scratch

Or if you have questions about the fix?

