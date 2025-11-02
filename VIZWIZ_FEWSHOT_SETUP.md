# VizWiz Few-Shot Object Detection with DeFRCN on GCP

Complete setup guide for reproducing VizWiz few-shot results following the OD-25ᵢ protocol.

## ✅ Setup Complete

All files have been created and configured:

- `vizwiz_folds.py` - Fold definitions (75 base / 25 novel per fold)
- `prepare_vizwiz_folds.py` - Annotation preparation script  
- `train_vizwiz_fewshot.sh` - Main training script with W&B logging
- `defrcn/data/register_vizwiz.py` - Dataset registration
- `configs/vizwiz_det_r101_base.yaml` - Base config with NUM_CLASSES=101

## 📊 Configuration

### Optimized Settings (matching Lambda Labs Co-DETR):
- **Base training**: 8,000 iterations (~8 epochs)
- **Few-shot training**: 2,500 iterations (~25 epochs)
- **K-shots**: 1, 10 (most informative comparison)
- **All 4 folds**: OD25_0, OD25_1, OD25_2, OD25_3
- **W&B logging**: Project `defrcn_vizwiz`

### Dataset Statistics:
```
OD25_0: 1,735 base train / 1,180 novel train
OD25_1: 2,275 base train / 640 novel train
OD25_2: 2,350 base train / 565 novel train
OD25_3: 1,470 base train / 1,445 novel train
```

## 🚀 Usage

### 1. Activate Environment

```bash
conda activate defrcn-cu121
cd ~/DeFRCN
```

### 2. Verify Setup

```bash
# Check that fold annotations were created
ls -la datasets/vizwiz/annotations/OD25_0/

# Test dataset registration
python -c "from defrcn.data import register_vizwiz; print('✅ Datasets registered')"

# Verify NUM_CLASSES=101
grep "NUM_CLASSES" configs/vizwiz_det_r101_base.yaml
```

### 3. Check Pretrained Weights

```bash
ls -la data/.pretrain_weights/ImageNetPretrained/MSRA/R-101.pkl
ls -la data/.pretrain_weights/ImageNetPretrained/torchvision/resnet101-5d3b4d8f.pth
```

If missing, download from:
- R-101.pkl: [Detectron2 Model Zoo](https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md)
- resnet101-5d3b4d8f.pth: [PyTorch Hub](https://download.pytorch.org/models/resnet101-5d3b4d8f.pth)

### 4. Configure Weights & Biases

```bash
# Login to W&B (one time)
wandb login

# Or set API key
export WANDB_API_KEY="your_key_here"
```

### 5. Start Training

```bash
# Run complete training pipeline (all 4 folds, k={1,10})
./train_vizwiz_fewshot.sh vizwiz_defrcn_experiment

# Or specify custom experiment name
./train_vizwiz_fewshot.sh my_experiment_name
```

### 6. Monitor Progress

- **W&B Dashboard**: https://wandb.ai/your-username/defrcn_vizwiz
- **Local logs**: `outputs/vizwiz_fewshot/`

## 📈 Expected Timeline

**Per Fold:**
- Base training (8,000 iter): ~3-4 hours (8 GPUs)
- 1-shot fine-tuning: ~1 hour
- 10-shot fine-tuning: ~1 hour
- **Subtotal**: ~5-6 hours per fold

**Total for 4 Folds**: ~20-24 hours

## 📁 Output Structure

```
outputs/vizwiz_fewshot/
├── OD25_0/
│   ├── base_training/
│   │   ├── model_final.pth
│   │   ├── model_reset_remove.pth
│   │   └── metrics.json
│   ├── novel_1shot/
│   │   ├── model_final.pth
│   │   └── metrics.json
│   └── novel_10shot/
│       ├── model_final.pth
│       └── metrics.json
├── OD25_1/
│   └── ...
├── OD25_2/
│   └── ...
└── OD25_3/
    └── ...
```

## 🔍 Troubleshooting

### Issue: "Dataset not found"
**Solution**: Check dataset registration
```bash
python -c "from detectron2.data import DatasetCatalog; print(DatasetCatalog.list())"
```

### Issue: "CUDA out of memory"
**Solution**: Reduce batch size in training script
```bash
# Edit train_vizwiz_fewshot.sh
# Change: IMS_PER_BATCH: 16
# To: IMS_PER_BATCH: 8
```

### Issue: "NUM_CLASSES mismatch"
**Solution**: Already fixed! NUM_CLASSES=101 to handle VizWiz category IDs 1-100

## 📊 Results Analysis

After training completes, results will be logged to W&B showing:
- **Base mAP**: Performance on 75 base categories
- **Novel mAP**: Performance on 25 novel categories (1-shot and 10-shot)
- **Per-fold results**: Individual performance for each fold
- **Averaged results**: Mean ± std across all 4 folds

## 🎯 Key Differences from Co-DETR

1. **Architecture**: DeFRCN uses Faster R-CNN vs Co-DETR's DETR-based model
2. **Training**: DeFRCN uses iterations vs Co-DETR's epochs
3. **Model Surgery**: DeFRCN requires explicit head removal/replacement
4. **PCB Module**: DeFRCN's Prototypical Calibration Block for few-shot

## 📚 References

- [DeFRCN Paper](https://arxiv.org/abs/2108.09017)
- [DeFRCN GitHub](https://github.com/er-muyue/DeFRCN)
- [VizWiz Dataset](https://vizwiz.org/)

## ✅ Pre-Flight Checklist

- [x] Fold annotations prepared (all 4 folds)
- [x] NUM_CLASSES set to 101
- [x] Dataset registration updated
- [x] Training script created with W&B logging
- [x] Optimized configuration (8 base epochs, 25 few-shot, k={1,10})
- [ ] Pretrained weights downloaded
- [ ] W&B configured
- [ ] GPU resources available (8 GPUs recommended)

Ready to train! 🚀

