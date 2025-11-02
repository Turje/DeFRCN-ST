#!/usr/bin/env python3
"""
Comprehensive DeFRCN Pre-Training Verification
Similar to Co-DETR verification - checks EVERYTHING before starting
"""

import os
import json
import sys
from pathlib import Path

print("=" * 80)
print("🔍 COMPREHENSIVE DeFRCN PRE-TRAINING VERIFICATION")
print("=" * 80)
print()

# Track issues
issues = []
warnings_list = []

# ============================================
# 1. GPU CHECK
# ============================================
print("1️⃣  GPU Availability Check")
print("-" * 80)
try:
    import torch
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"✅ Found {gpu_count} GPU(s)")
        for i in range(gpu_count):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
            mem = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"           Memory: {mem:.1f} GB")
    else:
        issues.append("No GPUs detected!")
        print("❌ No GPUs detected!")
except Exception as e:
    issues.append(f"GPU check failed: {e}")
    print(f"❌ GPU check failed: {e}")

print()

# ============================================
# 2. DATA STRUCTURE CHECK
# ============================================
print("2️⃣  Data Structure Check")
print("-" * 80)

base_data_path = Path("datasets/vizwiz/annotations")
required_folds = ["OD25_0", "OD25_1", "OD25_2", "OD25_3"]

if base_data_path.exists():
    print(f"✅ Base path exists: {base_data_path}")
    
    for fold in required_folds:
        fold_path = base_data_path / fold
        if fold_path.exists():
            files = list(fold_path.glob("*.json"))
            print(f"✅ {fold}: {len(files)} annotation files")
        else:
            issues.append(f"Missing fold: {fold}")
            print(f"❌ {fold}: Not found!")
else:
    issues.append(f"Data path not found: {base_data_path}")
    print(f"❌ Base path not found: {base_data_path}")

print()

# ============================================
# 3. ANNOTATION FILE VERIFICATION
# ============================================
print("3️⃣  Annotation File Class Count Verification")
print("-" * 80)

expected_classes = {
    "base": 75,
    "novel": 25,
    "total": 100
}

for fold in required_folds:
    print(f"\n{fold}:")
    
    # Check base training
    base_train = base_data_path / fold / "instances_train_base.json"
    if base_train.exists():
        with open(base_train) as f:
            data = json.load(f)
            cats = len(data['categories'])
            if cats == expected_classes['base']:
                print(f"  ✅ Base train: {cats} classes (correct)")
            else:
                issues.append(f"{fold} base train has {cats} classes, expected {expected_classes['base']}")
                print(f"  ❌ Base train: {cats} classes (expected {expected_classes['base']})")
    else:
        issues.append(f"{fold} base train file missing")
        print(f"  ❌ Base train file missing")
    
    # Check novel validation
    novel_val = base_data_path / fold / "instances_val_novel.json"
    if novel_val.exists():
        with open(novel_val) as f:
            data = json.load(f)
            cats = len(data['categories'])
            if cats == expected_classes['novel']:
                print(f"  ✅ Novel val: {cats} classes (correct)")
            else:
                warnings_list.append(f"{fold} novel val has {cats} classes, expected {expected_classes['novel']}")
                print(f"  ⚠️  Novel val: {cats} classes (expected {expected_classes['novel']})")
    else:
        issues.append(f"{fold} novel val file missing")
        print(f"  ❌ Novel val file missing")
    
    # Check few-shot files
    for k in [1, 3, 5, 10]:
        shot_file = base_data_path / fold / f"instances_train_novel_{k}shot.json"
        if shot_file.exists():
            print(f"  ✅ {k}-shot file exists")
        else:
            issues.append(f"{fold} {k}-shot file missing")
            print(f"  ❌ {k}-shot file missing")

print()

# ============================================
# 4. CONFIG FILE CHECK
# ============================================
print("4️⃣  Configuration File Check")
print("-" * 80)

configs_to_check = [
    ("configs/vizwiz_det_r101_base_balanced.yaml", 75, "base"),
    ("configs/vizwiz_det_r101_novel_1shot.yaml", 100, "novel"),
    ("configs/vizwiz_det_r101_novel_10shot.yaml", 100, "novel"),
]

for config_path, expected_classes, stage in configs_to_check:
    config_file = Path(config_path)
    if config_file.exists():
        with open(config_file) as f:
            content = f.read()
            # Look for NUM_CLASSES
            if "NUM_CLASSES:" in content:
                for line in content.split('\n'):
                    if 'NUM_CLASSES:' in line and not line.strip().startswith('#'):
                        # Extract number
                        num_classes = int(line.split(':')[1].strip().split('#')[0].strip())
                        if num_classes == expected_classes:
                            print(f"✅ {config_file.name}: NUM_CLASSES = {num_classes} (correct)")
                        else:
                            issues.append(f"{config_file.name} has NUM_CLASSES={num_classes}, expected {expected_classes}")
                            print(f"❌ {config_file.name}: NUM_CLASSES = {num_classes} (expected {expected_classes})")
                        break
            else:
                warnings_list.append(f"{config_file.name} doesn't have NUM_CLASSES set")
                print(f"⚠️  {config_file.name}: NUM_CLASSES not found")
    else:
        issues.append(f"Config file missing: {config_path}")
        print(f"❌ {config_file.name}: Not found")

print()

# ============================================
# 5. TRAINING SCRIPT CHECK
# ============================================
print("5️⃣  Training Script Check")
print("-" * 80)

train_script = Path("train_vizwiz_fewshot_codetr_2x.sh")
if train_script.exists():
    print(f"✅ Training script exists: {train_script}")
    with open(train_script) as f:
        content = f.read()
        # Check for all folds
        folds_in_script = [f for f in required_folds if f in content]
        if len(folds_in_script) == len(required_folds):
            print(f"✅ All {len(required_folds)} folds present in script")
        else:
            warnings_list.append(f"Only {len(folds_in_script)}/{len(required_folds)} folds in script")
            print(f"⚠️  Only {len(folds_in_script)}/{len(required_folds)} folds in script")
        
        # Check for W&B config
        if "WANDB_PROJECT" in content:
            print("✅ W&B project configured")
        else:
            warnings_list.append("W&B project not configured in script")
            print("⚠️  W&B project not configured")
else:
    issues.append("Training script missing")
    print(f"❌ Training script not found: {train_script}")

print()

# ============================================
# 6. PREVIOUS OUTPUTS CHECK
# ============================================
print("6️⃣  Previous Training Outputs Check")
print("-" * 80)

outputs_dir = Path("outputs_codetr_2x")
if outputs_dir.exists():
    subdirs = [d for d in outputs_dir.iterdir() if d.is_dir()]
    if subdirs:
        print(f"⚠️  Found {len(subdirs)} existing output directories")
        print("   These will be overwritten on restart:")
        for d in subdirs[:5]:  # Show first 5
            print(f"   - {d.name}")
        if len(subdirs) > 5:
            print(f"   ... and {len(subdirs) - 5} more")
    else:
        print("✅ Outputs directory is empty")
else:
    print("✅ No previous outputs directory")

print()

# ============================================
# 7. ENVIRONMENT CHECK
# ============================================
print("7️⃣  Environment Check")
print("-" * 80)

required_packages = {
    'torch': 'PyTorch',
    'detectron2': 'Detectron2',
    'wandb': 'Weights & Biases',
}

for package, name in required_packages.items():
    try:
        __import__(package)
        print(f"✅ {name} installed")
    except ImportError:
        issues.append(f"{name} not installed")
        print(f"❌ {name} not installed")

print()

# ============================================
# FINAL SUMMARY
# ============================================
print("=" * 80)
print("📊 VERIFICATION SUMMARY")
print("=" * 80)

if issues:
    print(f"\n❌ FOUND {len(issues)} CRITICAL ISSUES:")
    for i, issue in enumerate(issues, 1):
        print(f"   {i}. {issue}")
    print("\n⚠️  CANNOT START TRAINING - Fix issues above first!")
    sys.exit(1)
else:
    print("\n✅ NO CRITICAL ISSUES FOUND!")

if warnings_list:
    print(f"\n⚠️  {len(warnings_list)} Warning(s):")
    for i, warning in enumerate(warnings_list, 1):
        print(f"   {i}. {warning}")
    print("\n✅ Can proceed but review warnings")
else:
    print("\n✅ NO WARNINGS!")

print("\n" + "=" * 80)
print("✅ VERIFICATION COMPLETE - READY TO TRAIN!")
print("=" * 80)

