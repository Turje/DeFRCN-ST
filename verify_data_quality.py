"""
Comprehensive data quality verification for VizWiz Few-Shot training.

Checks for issues that could cause low mAP:
1. Class imbalance
2. Images without annotations
3. Annotation format issues
4. Category ID ranges
5. Bbox validity
6. Image-annotation mismatches
"""

import json
import os
from collections import Counter, defaultdict
import numpy as np

def analyze_fold(fold_name):
    """Analyze a single fold for data quality issues."""
    print(f"\n{'='*80}")
    print(f"ANALYZING FOLD: {fold_name}")
    print(f"{'='*80}")
    
    fold_dir = f"datasets/vizwiz/annotations/{fold_name}"
    
    # Check base training
    train_base_file = os.path.join(fold_dir, "instances_train_base.json")
    val_base_file = os.path.join(fold_dir, "instances_val_base.json")
    train_novel_file = os.path.join(fold_dir, "instances_train_novel.json")
    
    with open(train_base_file, 'r') as f:
        train_base = json.load(f)
    with open(val_base_file, 'r') as f:
        val_base = json.load(f)
    with open(train_novel_file, 'r') as f:
        train_novel = json.load(f)
    
    print(f"\n1️⃣ BASIC STATISTICS:")
    print(f"   Base train: {len(train_base['images'])} images, {len(train_base['annotations'])} anns, {len(train_base['categories'])} cats")
    print(f"   Base val: {len(val_base['images'])} images, {len(val_base['annotations'])} anns, {len(val_base['categories'])} cats")
    print(f"   Novel train: {len(train_novel['images'])} images, {len(train_novel['annotations'])} anns, {len(train_novel['categories'])} cats")
    
    # Check for images without annotations
    print(f"\n2️⃣ IMAGES WITHOUT ANNOTATIONS:")
    train_base_img_ids = set([img['id'] for img in train_base['images']])
    train_base_ann_ids = set([ann['image_id'] for ann in train_base['annotations']])
    no_ann_base = train_base_img_ids - train_base_ann_ids
    
    if len(no_ann_base) > 0:
        print(f"   ⚠️ Base train: {len(no_ann_base)} images WITHOUT annotations!")
    else:
        print(f"   ✅ Base train: All images have annotations")
    
    val_base_img_ids = set([img['id'] for img in val_base['images']])
    val_base_ann_ids = set([ann['image_id'] for ann in val_base['annotations']])
    no_ann_val = val_base_img_ids - val_base_ann_ids
    
    if len(no_ann_val) > 0:
        print(f"   ⚠️ Base val: {len(no_ann_val)} images WITHOUT annotations!")
    else:
        print(f"   ✅ Base val: All images have annotations")
    
    # Check class distribution (imbalance)
    print(f"\n3️⃣ CLASS DISTRIBUTION (Base Training):")
    cat_counts = Counter([ann['category_id'] for ann in train_base['annotations']])
    
    print(f"   Total categories: {len(cat_counts)}")
    print(f"   Min samples: {min(cat_counts.values())} (category {min(cat_counts, key=cat_counts.get)})")
    print(f"   Max samples: {max(cat_counts.values())} (category {max(cat_counts, key=cat_counts.get)})")
    print(f"   Mean samples: {np.mean(list(cat_counts.values())):.1f}")
    print(f"   Median samples: {np.median(list(cat_counts.values())):.1f}")
    
    # Check for severe imbalance
    imbalance_ratio = max(cat_counts.values()) / min(cat_counts.values())
    print(f"   Imbalance ratio: {imbalance_ratio:.1f}x")
    
    if imbalance_ratio > 100:
        print(f"   ⚠️ SEVERE class imbalance detected!")
    elif imbalance_ratio > 10:
        print(f"   ⚠️ Moderate class imbalance")
    else:
        print(f"   ✅ Reasonable class balance")
    
    # Show top 5 and bottom 5 classes
    sorted_cats = sorted(cat_counts.items(), key=lambda x: x[1], reverse=True)
    print(f"\n   Top 5 classes (most samples):")
    for cat_id, count in sorted_cats[:5]:
        cat_name = next((c['name'] for c in train_base['categories'] if c['id'] == cat_id), 'Unknown')
        print(f"     - Category {cat_id} ({cat_name}): {count} annotations")
    
    print(f"\n   Bottom 5 classes (least samples):")
    for cat_id, count in sorted_cats[-5:]:
        cat_name = next((c['name'] for c in train_base['categories'] if c['id'] == cat_id), 'Unknown')
        print(f"     - Category {cat_id} ({cat_name}): {count} annotations")
    
    # Check category ID range
    print(f"\n4️⃣ CATEGORY ID RANGE:")
    all_cat_ids = [c['id'] for c in train_base['categories']]
    print(f"   Category IDs: {min(all_cat_ids)} to {max(all_cat_ids)}")
    print(f"   Expected: 1 to 100 (VizWiz has IDs 1-100)")
    
    if max(all_cat_ids) > 100:
        print(f"   ⚠️ Category IDs exceed 100! This will cause issues with NUM_CLASSES=101")
    else:
        print(f"   ✅ Category IDs within valid range")
    
    # Check bbox validity
    print(f"\n5️⃣ BOUNDING BOX VALIDITY:")
    invalid_bboxes = []
    zero_area_bboxes = []
    
    for ann in train_base['annotations']:
        bbox = ann['bbox']
        if len(bbox) != 4:
            invalid_bboxes.append(ann['id'])
        elif bbox[2] <= 0 or bbox[3] <= 0:
            zero_area_bboxes.append(ann['id'])
    
    if len(invalid_bboxes) > 0:
        print(f"   ⚠️ {len(invalid_bboxes)} annotations with invalid bbox format!")
    else:
        print(f"   ✅ All bboxes have valid format [x, y, w, h]")
    
    if len(zero_area_bboxes) > 0:
        print(f"   ⚠️ {len(zero_area_bboxes)} annotations with zero/negative area!")
    else:
        print(f"   ✅ All bboxes have positive area")
    
    # Check for annotations per image distribution
    print(f"\n6️⃣ ANNOTATIONS PER IMAGE:")
    anns_per_img = defaultdict(int)
    for ann in train_base['annotations']:
        anns_per_img[ann['image_id']] += 1
    
    anns_counts = list(anns_per_img.values())
    print(f"   Min annotations/image: {min(anns_counts)}")
    print(f"   Max annotations/image: {max(anns_counts)}")
    print(f"   Mean annotations/image: {np.mean(anns_counts):.2f}")
    print(f"   Median annotations/image: {np.median(anns_counts):.0f}")
    
    # Images with very few annotations
    sparse_images = sum(1 for count in anns_counts if count == 1)
    print(f"   Images with only 1 annotation: {sparse_images} ({100*sparse_images/len(anns_counts):.1f}%)")
    
    # Check if categories overlap between base and novel (data leakage)
    print(f"\n7️⃣ DATA LEAKAGE CHECK:")
    base_cat_ids = set([c['id'] for c in train_base['categories']])
    novel_cat_ids = set([c['id'] for c in train_novel['categories']])
    overlap = base_cat_ids & novel_cat_ids
    
    if len(overlap) > 0:
        print(f"   ⚠️ DATA LEAKAGE: {len(overlap)} categories appear in BOTH base and novel!")
        print(f"      Overlapping IDs: {sorted(list(overlap))[:10]}")
    else:
        print(f"   ✅ No category overlap between base and novel")
    
    # Check annotation format
    print(f"\n8️⃣ ANNOTATION FORMAT:")
    sample_ann = train_base['annotations'][0]
    required_fields = ['id', 'image_id', 'category_id', 'bbox', 'area']
    
    missing_fields = [f for f in required_fields if f not in sample_ann]
    if missing_fields:
        print(f"   ⚠️ Missing required fields: {missing_fields}")
    else:
        print(f"   ✅ All required fields present")
    
    print(f"   Sample annotation: {sample_ann}")
    
    return {
        'images_no_ann': len(no_ann_base),
        'imbalance_ratio': imbalance_ratio,
        'min_samples': min(cat_counts.values()),
        'max_samples': max(cat_counts.values()),
        'data_leakage': len(overlap)
    }

def main():
    print("="*80)
    print("VIZWIZ FEW-SHOT DATA QUALITY VERIFICATION")
    print("="*80)
    
    issues = []
    
    for fold in ["OD25_0", "OD25_1", "OD25_2", "OD25_3"]:
        stats = analyze_fold(fold)
        
        # Collect issues
        if stats['images_no_ann'] > 0:
            issues.append(f"{fold}: {stats['images_no_ann']} images without annotations")
        if stats['imbalance_ratio'] > 100:
            issues.append(f"{fold}: Severe class imbalance ({stats['imbalance_ratio']:.0f}x)")
        if stats['min_samples'] < 5:
            issues.append(f"{fold}: Some classes have <5 samples")
        if stats['data_leakage'] > 0:
            issues.append(f"{fold}: Category overlap between base/novel")
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    if len(issues) == 0:
        print("\n✅ No major data quality issues detected!")
        print("   Dataset appears to be correctly prepared for few-shot training.")
    else:
        print(f"\n⚠️ Found {len(issues)} potential issues:")
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")
    
    print(f"\n{'='*80}")
    print("RECOMMENDATIONS")
    print(f"{'='*80}")
    print("""
1. Class Imbalance:
   - DeFRCN handles this better than YOLO due to Faster R-CNN's RPN
   - Consider using focal loss or class-balanced sampling if issues persist
   
2. Low mAP Causes:
   - Check if NUM_CLASSES=101 is set correctly
   - Verify pretrained weights are loaded
   - Check learning rate (too high can cause instability)
   - Ensure proper data augmentation
   
3. Comparison with YOLO:
   - YOLO on 10 classes with 0.5 mAP is good
   - DeFRCN on 75 classes will naturally have lower initial mAP
   - Few-shot performance improves significantly with proper base training
   
4. If mAP is still low:
   - Check training logs for loss convergence
   - Verify GPU memory is sufficient
   - Try reducing learning rate by 10x
   - Ensure W&B is logging correctly
    """)

if __name__ == "__main__":
    main()

