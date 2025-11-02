# DeFRCN Critical Fixes Applied

## Issues Found:

### 1. ✅ **SPARSE CATEGORY IDs** (CRITICAL - FIXED)
- **Problem:** VizWiz has non-contiguous category IDs (1, 2, 3, 5, 6, ... instead of 1, 2, 3, 4, 5, ...)
- **Impact:** Model predictions were being evaluated against wrong class IDs
- **Result:** Base mAP only 6.59%, 1-shot mAP only 2.66%
- **Fix:** Added explicit `thing_dataset_id_to_contiguous_id` mapping in dataset registration

### 2. ✅ **PCB_MODELPATH Missing** (FIXED)
- **Problem:** Prototypical Calibration Block had empty model path
- **Impact:** PCB couldn't load ImageNet features for calibration
- **Fix:** Set `PCB_MODELPATH: "imagenet_pretrain/torchvision/resnet101-5d3b4d8f.pth"` in all novel configs

### 3. ✅ **Model num_classes** (Already Correct)
- Base model: 75 classes ✅
- Novel models: 100 classes ✅
- No changes needed

## Files Modified:

1. **configs/vizwiz_det_r101_novel_*.yaml** (all k-shots)
   - Added PCB_MODELPATH pointing to ImageNet weights

2. **defrcn/data/datasets/vizwiz.py**
   - Replaced with vizwiz_fixed_v2.py
   - Added explicit sparse->contiguous ID mapping
   - Function: `thing_dataset_id_to_contiguous_id={1: 0, 2: 1, 3: 2, 5: 3, ...}`

## Expected Results After Fix:

### Before Fix:
- Base mAP: 6.59%
- 1-shot mAP: 2.66%

### After Fix (Expected):
- Base mAP: 15-30% (2-4x improvement)
- 1-shot mAP: 5-15% (2-6x improvement)
- 10-shot mAP: 20-40%

## Training Status:

Current training needs to be restarted to apply these fixes.
All completed work (OD25_0 base, 1-shot) should be redone with fixed evaluation.

## Commands to Restart:

```bash
cd /home/turje87/DeFRCN

# Stop current training
pkill -f "python.*main.py"

# Clean old outputs (backup first)
mv outputs_defrcn outputs_defrcn_OLD_BEFORE_FIX_$(date +%Y%m%d)

# Restart with fixes
nohup bash train_vizwiz_fewshot_codetr_2x.sh > training_gcp_DFRCN_FIXED.log 2>&1 &

# Monitor
python3 monitor_defrcn_comprehensive.py
```

## Timeline:

- OD25_0: ~2-3 hours (base + 4 k-shots)
- All 4 folds: ~8-10 hours total

Expected completion: +10 hours from restart

