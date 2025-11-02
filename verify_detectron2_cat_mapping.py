#!/usr/bin/env python3
"""
Verify if Detectron2 automatically handles sparse category IDs
"""
import json
import sys
from pathlib import Path

# Add Detectron2 to path
sys.path.insert(0, '/home/turje87/DeFRCN')

from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog
import detectron2.data.datasets.coco as coco_data

print("="*70)
print("DETECTRON2 SPARSE CATEGORY ID VERIFICATION")
print("="*70)

# Load annotation file
ann_file = Path("/home/turje87/DeFRCN/datasets/vizwiz/annotations/OD25_0/instances_train_base.json")

with open(ann_file) as f:
    data = json.load(f)

categories = sorted(data['categories'], key=lambda x: x['id'])
cat_ids = [c['id'] for c in categories]

print(f"\n1️⃣  ANNOTATION FILE:")
print(f"   Categories: {len(categories)}")
print(f"   Category IDs: {cat_ids[:20]}... (showing first 20)")
print(f"   Has gaps: {cat_ids != list(range(min(cat_ids), max(cat_ids)+1))}")

# Now check what Detectron2 does
print(f"\n2️⃣  DETECTRON2 DATASET LOADING:")
print("-"*70)

# Try to load with Detectron2
try:
    # Register if not already registered
    dataset_name = "vizwiz_verify_test"
    if dataset_name in DatasetCatalog:
        DatasetCatalog.remove(dataset_name)
        MetadataCatalog.remove(dataset_name)
    
    register_coco_instances(
        dataset_name,
        {},
        str(ann_file),
        "/home/turje87/DeFRCN/datasets/vizwiz/images/train"
    )
    
    # Get dataset
    dataset_dicts = DatasetCatalog.get(dataset_name)
    metadata = MetadataCatalog.get(dataset_name)
    
    print(f"   ✅ Loaded {len(dataset_dicts)} images")
    print(f"   Thing classes: {len(metadata.thing_classes)}")
    
    # Check if there's a mapping
    if hasattr(metadata, 'thing_dataset_id_to_contiguous_id'):
        mapping = metadata.thing_dataset_id_to_contiguous_id
        print(f"\n   ✅ FOUND: thing_dataset_id_to_contiguous_id mapping!")
        print(f"   Mapping sample:")
        for k, v in list(mapping.items())[:15]:
            print(f"      Category ID {k} → Contiguous {v}")
        
        # Verify the mapping
        print(f"\n   🔍 VERIFICATION:")
        if mapping.get(1) == 0 and mapping.get(2) == 1 and mapping.get(12) == 10:
            print(f"      ✅ Mapping is CORRECT!")
            print(f"         ID 1 → 0, ID 2 → 1, ID 12 → 10 (skipping gap at 11)")
        else:
            print(f"      ❌ Mapping seems incorrect")
            print(f"         ID 1 → {mapping.get(1)}")
            print(f"         ID 2 → {mapping.get(2)}")
            print(f"         ID 12 → {mapping.get(12)}")
    else:
        print(f"\n   ❌ NO MAPPING FOUND!")
        print(f"      thing_dataset_id_to_contiguous_id does NOT exist")
    
    # Check actual annotations in dataset_dicts
    print(f"\n3️⃣  CHECKING LOADED ANNOTATIONS:")
    print("-"*70)
    
    sample_img = dataset_dicts[0]
    if sample_img.get('annotations'):
        sample_ann = sample_img['annotations'][0]
        print(f"   Sample annotation keys: {sample_ann.keys()}")
        print(f"   category_id: {sample_ann.get('category_id', 'N/A')}")
        
        # Collect all category_ids from loaded data
        all_cat_ids = set()
        for img_dict in dataset_dicts[:100]:
            for ann in img_dict.get('annotations', []):
                all_cat_ids.add(ann['category_id'])
        
        all_cat_ids = sorted(all_cat_ids)
        print(f"\n   Category IDs in loaded data: {all_cat_ids[:15]}...")
        
        # Check if they're contiguous
        expected = list(range(len(all_cat_ids)))
        if all_cat_ids == expected or all_cat_ids == [x+1 for x in expected]:
            print(f"   ✅ IDs are CONTIGUOUS: {min(all_cat_ids)}-{max(all_cat_ids)}")
        else:
            print(f"   ❌ IDs are SPARSE: {min(all_cat_ids)}-{max(all_cat_ids)} with gaps")

except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("CONCLUSION:")
print("="*70)
print("If 'thing_dataset_id_to_contiguous_id' exists and is correct,")
print("then Detectron2 SHOULD handle sparse IDs automatically.")
print("Otherwise, the sparse IDs are causing the 0% mAP issue.")
print("="*70)

