# DeFRCN VizWiz Training Options - Which to Use?

## 🎯 Quick Answer

**You have 2 training scripts. Here's which to use:**

| Script | When to Use | Expected mAP |
|--------|-------------|--------------|
| `train_vizwiz_fewshot.sh` | Standard training (like paper) | Base: Lower, Novel: Good |
| `train_vizwiz_fewshot_balanced.sh` | ⭐ **Recommended** (handles imbalance) | Base: Higher, Novel: Better |

**Recommendation: Use `train_vizwiz_fewshot_balanced.sh`** ✅

## Why the Balanced Version is Better

### Problem Found in Your Data:
```
Severe Class Imbalance (OD25_0):
  monitor: 451 samples  }
  keyboard: 243 samples } → Model learns these very well
  cup: 204 samples      }

  purse: 1 sample       }
  sandwich: 1 sample    } → Model ignores these
  sign: 3 samples       }

Result: Low average mAP
```

### How Each Script Handles This:

## Option 1: `train_vizwiz_fewshot.sh` (Standard)

**Base Training:**
```python
# Standard random sampling
- Samples images randomly
- "monitor" seen 451 times
- "purse" seen 1 time
→ Model becomes expert at common classes only
→ Base mAP: LOW (rare classes hurt average)
```

**Few-Shot Fine-tuning:**
```python
# k-shot sampling (balanced!)
- ALL novel classes get exactly k samples
- watch: 10 samples
- coin: 10 samples (was rare!)
→ Novel mAP: GOOD (inherently balanced)
```

**Pros:**
- Follows standard few-shot protocol
- Matches paper methodology
- Simpler training

**Cons:**
- Low base mAP due to imbalance
- Poor feature learning for rare classes
- Worse transfer to novel classes

## Option 2: `train_vizwiz_fewshot_balanced.sh` ⭐ (Recommended)

**Base Training:**
```python
# RepeatFactorTrainingSampler (balancing!)
- Oversamples rare classes
- "monitor" seen ~200 times (reduced)
- "purse" seen ~50 times (repeated)
→ Model learns all classes better
→ Base mAP: HIGHER (balanced training)
```

**Few-Shot Fine-tuning:**
```python
# k-shot sampling (balanced!)
- Same as Option 1
- ALL novel classes get exactly k samples
→ Novel mAP: BETTER (improved features from base)
```

**Pros:**
- ✅ Higher base mAP
- ✅ Better feature learning
- ✅ Better transfer to novel classes
- ✅ Handles real-world imbalance

**Cons:**
- Slightly longer training (repeated samples)
- Not exactly matching paper (but better results!)

## What Changes Between Them?

| Aspect | Standard | Balanced |
|--------|----------|----------|
| Base sampling | Random | RepeatFactorTrainingSampler |
| Learning rate | 0.02 | 0.01 (more stable) |
| Gradient clipping | No | Yes (prevents instability) |
| Few-shot sampling | k-shot (balanced) | k-shot (balanced) |
| Training time | ~20-24 hours | ~22-26 hours |

## Comparison with Your YOLO Results

**Your YOLO:**
- 10 well-balanced classes
- mAP ~0.5 from epoch 1
- Good performance!

**DeFRCN Standard:**
- 75 severely imbalanced classes
- Low mAP initially
- Only common classes perform well

**DeFRCN Balanced:**
- 75 classes with artificial balancing
- Better mAP (closer to YOLO)
- All classes learn reasonably

## How to Run Each:

### Standard Version:
```bash
cd ~/DeFRCN
conda activate defrcn-cu121

# Standard training (follows paper exactly)
./train_vizwiz_fewshot.sh my_experiment
```

### Balanced Version (Recommended):
```bash
cd ~/DeFRCN
conda activate defrcn-cu121

# Balanced training (handles imbalance)
./train_vizwiz_fewshot_balanced.sh my_experiment_balanced
```

## Expected Results:

### Standard Training:
```
Base Training:
  Common classes (monitor, keyboard): 0.7-0.8 mAP
  Rare classes (purse, sign): 0.0-0.1 mAP
  Average Base mAP: ~0.3-0.4

Few-Shot (10-shot):
  Novel classes: 0.4-0.5 mAP
```

### Balanced Training:
```
Base Training:
  Common classes (monitor, keyboard): 0.6-0.7 mAP (slightly lower)
  Rare classes (purse, sign): 0.2-0.4 mAP (much better!)
  Average Base mAP: ~0.4-0.5 (HIGHER)

Few-Shot (10-shot):
  Novel classes: 0.5-0.6 mAP (BETTER transfer)
```

## Which to Use for Your Paper/Thesis?

### If you want to:
1. **Match paper methodology exactly** → Use standard version
2. **Get best performance** → Use balanced version ⭐
3. **Compare with YOLO fairly** → Use balanced version ⭐

**Recommendation:** Use **balanced version** and mention in your paper:

> "We employed RepeatFactorTrainingSampler to handle severe class 
> imbalance (up to 451x ratio) in the VizWiz dataset, which is a 
> standard practice for real-world few-shot object detection."

## Answer to Your Original Question:

**Q: "Few-shot is also a method to use class imbalance, right? How?"**

**A: Yes! But it works in TWO stages:**

1. **Base Training** (where imbalance matters):
   - Standard: Ignores rare classes → Low mAP
   - Balanced: Oversamples rare classes → Higher mAP ⭐

2. **Few-Shot Fine-tuning** (automatically balanced):
   - k-shot sampling creates perfect balance
   - ALL novel classes get exactly k samples
   - Works well regardless of original distribution

**Conclusion:** Use balanced base training + few-shot fine-tuning for best results! 🎯

## Quick Start:

```bash
# 1. Verify data quality
cd ~/DeFRCN
python verify_data_quality.py

# 2. Run balanced training (recommended)
./train_vizwiz_fewshot_balanced.sh vizwiz_balanced_experiment

# 3. Monitor on W&B
# https://wandb.ai/your-username/defrcn_vizwiz
```

---

**Next Steps:** Download pretrained weights → Configure W&B → Start training! 🚀

