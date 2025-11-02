# DeFRCN Training Time Estimates (Single GPU - 23GB)

## 📊 Timing Breakdown

### Assumptions (Based on Single GPU with Batch Size 8):
- **Base training**: ~0.8 seconds per iteration
- **Few-shot training**: ~0.5 seconds per iteration
- **Model surgery**: ~1 minute
- **Evaluation**: ~5-10 minutes per checkpoint

---

## Option 1: ULTRA FAST ⚡ (`train_vizwiz_fast.sh`)

### Configuration:
- Base: 2,000 iterations
- Novel: 500 iterations  
- Folds: 1 (OD25_0)
- K-shots: 10 only

### Time per Fold:
```
Base training:    2,000 iter × 0.8 sec = 1,600 sec = 27 minutes
Base evaluation:                         = 5 minutes
Model surgery:                           = 1 minute
Novel training:   500 iter × 0.5 sec   = 250 sec = 4 minutes
Novel evaluation:                        = 3 minutes
─────────────────────────────────────────────────────────
Total per fold:                          ≈ 40 minutes
```

### Total for 1 fold: **~40 minutes to 1 hour** ⚡

**If you run all 4 folds:** ~2.5-4 hours

---

## Option 2: MODERATE ⚡⚡ (Best for Thesis)

### Configuration:
- Base: 10,000 iterations
- Novel: 2,000 iterations
- Folds: 4 (OD25_0 to OD25_3)
- K-shots: 1, 10

### Time per Fold:
```
Base training:    10,000 iter × 0.8 sec = 8,000 sec = 2.2 hours
Base evaluation:                          = 10 minutes
Model surgery:                            = 1 minute
Novel 1-shot:     2,000 iter × 0.5 sec  = 1,000 sec = 17 minutes
Novel 1-shot eval:                        = 5 minutes
Novel 10-shot:    2,000 iter × 0.5 sec  = 1,000 sec = 17 minutes
Novel 10-shot eval:                       = 5 minutes
─────────────────────────────────────────────────────────
Total per fold:                           ≈ 3 hours
```

### Total for 4 folds: **~12 hours** ⚡⚡

---

## Option 3: PAPER STANDARD ⚡⚡⚡ (`train_vizwiz_fewshot_balanced.sh`)

### Configuration:
- Base: 8,000 iterations (already created)
- Novel: 2,500 iterations
- Folds: 4
- K-shots: 1, 10

### Time per Fold:
```
Base training:    8,000 iter × 0.8 sec  = 6,400 sec = 1.8 hours
Base evaluation:                          = 10 minutes
Model surgery:                            = 1 minute
Novel 1-shot:     2,500 iter × 0.5 sec  = 1,250 sec = 21 minutes
Novel 1-shot eval:                        = 5 minutes
Novel 10-shot:    2,500 iter × 0.5 sec  = 1,250 sec = 21 minutes
Novel 10-shot eval:                       = 5 minutes
─────────────────────────────────────────────────────────
Total per fold:                           ≈ 2.7 hours
```

### Total for 4 folds: **~11 hours** ⚡⚡⚡

---

## Option 4: DEFRCN ORIGINAL (Not Recommended)

### Configuration:
- Base: 110,000 iterations
- Novel: 10,000 iterations
- Folds: 4
- K-shots: 1, 10

### Time per Fold:
```
Base training:    110,000 iter × 0.8 sec = 88,000 sec = 24.4 hours
Base evaluation:                           = 20 minutes
Model surgery:                             = 1 minute
Novel 1-shot:     10,000 iter × 0.5 sec  = 5,000 sec = 1.4 hours
Novel 1-shot eval:                         = 10 minutes
Novel 10-shot:    10,000 iter × 0.5 sec  = 5,000 sec = 1.4 hours
Novel 10-shot eval:                        = 10 minutes
─────────────────────────────────────────────────────────
Total per fold:                            ≈ 28 hours
```

### Total for 4 folds: **~112 hours (4.7 days)** 😱

---

## 📊 Quick Comparison Table

| Option | Base Iter | Novel Iter | Time (1 fold) | Time (4 folds) | mAP Expected | Recommended |
|--------|-----------|------------|---------------|----------------|--------------|-------------|
| **ULTRA FAST** | 2,000 | 500 | **40 min** | **2.5 hours** | 0.2-0.4 | Testing only |
| **MODERATE** | 10,000 | 2,000 | **3 hours** | **12 hours** | 0.4-0.5 | ✅ **YES - Thesis** |
| **PAPER STD** | 8,000 | 2,500 | **2.7 hours** | **11 hours** | 0.5-0.6 | Yes - Final |
| **DEFRCN ORIG** | 110,000 | 10,000 | **28 hours** | **112 hours** | 0.6+ | ❌ Too long |

---

## 🎯 My Recommendation

### For Quick Testing (Today):
**Use ULTRA FAST** - Get results in **~1 hour** (1 fold)
```bash
./train_vizwiz_fast.sh test_run
# Time: 40 min - 1 hour
# Good for: Verifying pipeline works
```

### For Your Thesis (This Week):
**Use PAPER STANDARD** - Get results in **~11 hours** (4 folds)
```bash
./train_vizwiz_fewshot_balanced.sh thesis_results
# Time: ~11 hours (overnight run)
# Good for: Final thesis results
# W&B: defrcn_gcp_cursor
```

**Why not MODERATE?** 
- PAPER STANDARD (8k iter) is actually FASTER than MODERATE (10k iter)!
- Already created and tested
- Better documented
- Matches Co-DETR setup from Lambda

---

## ⏰ Realistic Timeline

### Today (Testing):
```
Hour 0:00 - Start ULTRA FAST training
Hour 0:40 - See first results
Hour 1:00 - Training complete, verify mAP > 0.2
```

### Tonight (Final Results):
```
Hour 0:00 - Start PAPER STANDARD training (before bed)
Hour 2:30 - Fold 1 complete
Hour 5:00 - Fold 2 complete
Hour 7:30 - Fold 3 complete
Hour 11:00 - Fold 4 complete, ALL DONE! ✅
```

---

## 💰 Cost Estimate (If on GCP with GPU)

Assuming GCP costs ~$2-3/hour for 1 GPU:

| Option | Training Time | Cost Estimate |
|--------|---------------|---------------|
| ULTRA FAST | 1 hour | ~$2-3 |
| PAPER STD (4 folds) | 11 hours | ~$22-33 |
| MODERATE (4 folds) | 12 hours | ~$24-36 |
| DEFRCN ORIG (4 folds) | 112 hours | ~$224-336 |

---

## 🎬 What to Run NOW

### Step 1: Quick Test (Optional - 1 hour)
```bash
cd ~/DeFRCN
conda activate defrcn-cu121
./train_vizwiz_fast.sh quick_test

# Verify it works, check W&B
```

### Step 2: Full Training (Recommended - 11 hours)
```bash
cd ~/DeFRCN
conda activate defrcn-cu121

# Run overnight
nohup ./train_vizwiz_fewshot_balanced.sh final_results > training.log 2>&1 &

# Monitor
tail -f training.log
```

---

## ✅ Summary

**Answer to "How many hours will it train?"**

- **Testing**: 40 minutes (1 fold, ultra fast)
- **Thesis results**: **~11 hours** (4 folds, paper standard) ⭐
- **Full DeFRCN**: 112 hours (not recommended)

**Recommendation**: Run the **PAPER STANDARD** (~11 hours) tonight!

