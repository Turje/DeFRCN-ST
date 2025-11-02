#!/usr/bin/env python3
"""
Comprehensive annotation checker for VizWiz OD25 dataset
"""
import json
from pathlib import Path
from collections import Counter, defaultdict

def check_annotation_file(ann_file):
    """Comprehensive checks on a single annotation file"""
    print(f"\n{'='*70}")
    print(f"Checking: {ann_file.name}")
    print('='*70)
    
    if not ann_file.exists():
        print(f"❌ File does not exist!")
        return None
    
    with open(ann_file, 'r') as f:
        data = json.load(f)
    
    # Basic structure
    print(f"\n1️⃣  FILE STRUCTURE:")
    print(f"   Images: {len(data.get('images', []))}")
    print(f"   Annotations: {len(data.get('annotations', []))}")
    print(f"   Categories: {len(data.get('categories', []))}")
    
    # Category analysis
    categories = data.get('categories', [])
    cat_ids = [c['id'] for c in categories]
    cat_names = [c['name'] for c in categories]
    
    print(f"\n2️⃣  CATEGORY IDs:")
    print(f"   Min ID: {min(cat_ids) if cat_ids else 'N/A'}")
    print(f"   Max ID: {max(cat_ids) if cat_ids else 'N/A'}")
    print(f"   Total unique: {len(set(cat_ids))}")
    print(f"   Sequential: {cat_ids == list(range(min(cat_ids), max(cat_ids)+1))}")
    
    # Check for gaps
    if cat_ids:
        expected = set(range(min(cat_ids), max(cat_ids)+1))
        actual = set(cat_ids)
        gaps = expected - actual
        if gaps:
            print(f"   ⚠️  GAPS in IDs: {sorted(gaps)}")
        else:
            print(f"   ✅ No gaps in category IDs")
    
    # Category distribution in annotations
    annotations = data.get('annotations', [])
    ann_cat_ids = [ann['category_id'] for ann in annotations]
    cat_counter = Counter(ann_cat_ids)
    
    print(f"\n3️⃣  ANNOTATION DISTRIBUTION:")
    print(f"   Unique categories with annotations: {len(cat_counter)}")
    print(f"   Min annotations per category: {min(cat_counter.values()) if cat_counter else 0}")
    print(f"   Max annotations per category: {max(cat_counter.values()) if cat_counter else 0}")
    print(f"   Avg annotations per category: {sum(cat_counter.values())/len(cat_counter) if cat_counter else 0:.1f}")
    
    # Check for categories without annotations
    cats_with_anns = set(ann_cat_ids)
    cats_defined = set(cat_ids)
    cats_without_anns = cats_defined - cats_with_anns
    
    if cats_without_anns:
        print(f"   ⚠️  Categories with 0 annotations: {len(cats_without_anns)}")
        print(f"      IDs: {sorted(list(cats_without_anns))[:20]}")
    else:
        print(f"   ✅ All categories have annotations")
    
    # Check for annotations with undefined categories
    undefined_cats = cats_with_anns - cats_defined
    if undefined_cats:
        print(f"   ❌ Annotations with UNDEFINED categories: {len(undefined_cats)}")
        print(f"      IDs: {sorted(list(undefined_cats))[:20]}")
        print(f"      THIS IS A CRITICAL BUG!")
    else:
        print(f"   ✅ All annotation category_ids are defined")
    
    # Bounding box checks
    print(f"\n4️⃣  BOUNDING BOX CHECKS:")
    invalid_bboxes = 0
    bbox_formats = set()
    
    for ann in annotations[:100]:  # Sample first 100
        bbox = ann.get('bbox', [])
        if len(bbox) == 4:
            x, y, w, h = bbox
            if w <= 0 or h <= 0:
                invalid_bboxes += 1
            bbox_formats.add(f"format_xywh")
        else:
            bbox_formats.add(f"format_invalid_len{len(bbox)}")
    
    print(f"   Sample size: 100 annotations")
    print(f"   Bbox format: {bbox_formats}")
    if invalid_bboxes > 0:
        print(f"   ⚠️  Invalid bboxes (w<=0 or h<=0): {invalid_bboxes}/100")
    else:
        print(f"   ✅ All sampled bboxes valid")
    
    # Image ID checks
    print(f"\n5️⃣  IMAGE ID CHECKS:")
    image_ids = set(img['id'] for img in data.get('images', []))
    ann_image_ids = set(ann['image_id'] for ann in annotations)
    
    print(f"   Unique image IDs: {len(image_ids)}")
    print(f"   Images with annotations: {len(ann_image_ids)}")
    
    orphan_anns = ann_image_ids - image_ids
    if orphan_anns:
        print(f"   ❌ Annotations referencing missing images: {len(orphan_anns)}")
    else:
        print(f"   ✅ All annotations reference valid images")
    
    images_no_anns = image_ids - ann_image_ids
    if images_no_anns:
        print(f"   ⚠️  Images without annotations: {len(images_no_anns)}")
    
    # Print first few categories for reference
    print(f"\n6️⃣  SAMPLE CATEGORIES (first 10):")
    for cat in categories[:10]:
        count = cat_counter.get(cat['id'], 0)
        print(f"   ID {cat['id']:3d}: {cat['name']:30s} ({count:4d} annotations)")
    
    return {
        'total_images': len(data.get('images', [])),
        'total_annotations': len(annotations),
        'total_categories': len(categories),
        'min_cat_id': min(cat_ids) if cat_ids else None,
        'max_cat_id': max(cat_ids) if cat_ids else None,
        'has_gaps': len(gaps) > 0 if cat_ids else False,
        'cats_without_anns': len(cats_without_anns),
        'undefined_cats_in_anns': len(undefined_cats),
        'critical_issues': len(undefined_cats) > 0 or len(orphan_anns) > 0
    }

# Main execution
if __name__ == "__main__":
    print("="*70)
    print("COMPREHENSIVE ANNOTATION CHECK - VizWiz OD25")
    print("="*70)
    
    base_path = Path("/home/turje87/DeFRCN/datasets/vizwiz/annotations")
    
    # Check OD25_0 (currently training)
    fold = "OD25_0"
    fold_path = base_path / fold
    
    print(f"\n{'#'*70}")
    print(f"# FOLD: {fold}")
    print(f"{'#'*70}")
    
    results = {}
    
    # Check base annotations
    for split in ['train', 'val']:
        ann_file = fold_path / f"instances_{split}_base.json"
        if ann_file.exists():
            result = check_annotation_file(ann_file)
            results[f'{fold}_{split}_base'] = result
    
    # Check novel annotations
    for shot in [1, 10]:
        for split in ['train', 'val']:
            if split == 'train':
                ann_file = fold_path / f"instances_train_novel_{shot}shot.json"
            else:
                ann_file = fold_path / f"instances_val_novel.json"
            
            if ann_file.exists() and f'{fold}_{split}_novel' not in results:
                result = check_annotation_file(ann_file)
                results[f'{fold}_{split}_novel'] = result
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print('='*70)
    
    critical_issues = []
    for key, result in results.items():
        if result and result.get('critical_issues'):
            critical_issues.append(key)
            print(f"❌ {key}: CRITICAL ISSUES FOUND!")
        elif result:
            print(f"✅ {key}: OK (cats: {result['total_categories']}, anns: {result['total_annotations']})")
    
    if critical_issues:
        print(f"\n🚨 CRITICAL ISSUES IN: {', '.join(critical_issues)}")
        print(f"   These need to be fixed immediately!")
    else:
        print(f"\n✅ No critical issues found in annotation files")

