"""
Prepare VizWiz Few-Shot Dataset Splits for DeFRCN

This script creates base and novel splits for each fold following the OD-25ᵢ protocol.
It generates:
- Base training/validation sets (75 categories)
- Novel training/validation sets (25 categories)
- K-shot novel training sets (k ∈ {1, 3, 5, 10})
"""

import json
import os
import random
from collections import defaultdict
from vizwiz_folds import FOLDS, ALL_CATEGORIES

def load_annotations(ann_file):
    """Load COCO-format annotations."""
    print(f"Loading annotations from {ann_file}")
    with open(ann_file, 'r') as f:
        data = json.load(f)
    return data

def filter_annotations_by_categories(data, category_ids, keep_images_with_mixed=False):
    """
    Filter annotations to only include specified categories.
    
    Args:
        data: COCO-format annotation dict
        category_ids: List of category IDs to keep
        keep_images_with_mixed: If False, exclude images that have ANY annotations 
                               from categories NOT in category_ids (prevents data leakage)
    
    Returns:
        Filtered annotation dict
    """
    category_ids_set = set(category_ids)
    
    # Filter categories
    new_categories = [cat for cat in data['categories'] if cat['id'] in category_ids_set]
    
    # Filter annotations
    new_annotations = [ann for ann in data['annotations'] if ann['category_id'] in category_ids_set]
    
    # Get images that have annotations in our category set
    valid_image_ids = set([ann['image_id'] for ann in new_annotations])
    
    # If we want to prevent data leakage, exclude images that have ANY annotations 
    # from categories outside our set
    if not keep_images_with_mixed:
        # Find all images that have annotations from OTHER categories
        contaminated_image_ids = set()
        for ann in data['annotations']:
            if ann['category_id'] not in category_ids_set:
                contaminated_image_ids.add(ann['image_id'])
        
        # Remove contaminated images
        valid_image_ids = valid_image_ids - contaminated_image_ids
        
        # Filter annotations to only include valid images
        new_annotations = [ann for ann in new_annotations if ann['image_id'] in valid_image_ids]
    
    # Filter images
    new_images = [img for img in data['images'] if img['id'] in valid_image_ids]
    
    # Create new annotation dict
    filtered_data = {
        'images': new_images,
        'annotations': new_annotations,
        'categories': new_categories
    }
    
    return filtered_data

def create_k_shot_dataset(data, k, seed=0):
    """
    Create a k-shot dataset by sampling k images per category.
    
    Args:
        data: COCO-format annotation dict
        k: Number of shots per category
        seed: Random seed for reproducibility
    
    Returns:
        K-shot filtered annotation dict
    """
    random.seed(seed)
    
    # Group images by category
    category_to_images = defaultdict(set)
    for ann in data['annotations']:
        category_to_images[ann['category_id']].add(ann['image_id'])
    
    # Sample k images per category
    selected_image_ids = set()
    for cat_id, image_ids in category_to_images.items():
        image_list = list(image_ids)
        if len(image_list) < k:
            print(f"Warning: Category {cat_id} only has {len(image_list)} images, using all")
            selected_image_ids.update(image_list)
        else:
            sampled = random.sample(image_list, k)
            selected_image_ids.update(sampled)
    
    # Filter to selected images
    new_images = [img for img in data['images'] if img['id'] in selected_image_ids]
    new_annotations = [ann for ann in data['annotations'] if ann['image_id'] in selected_image_ids]
    
    return {
        'images': new_images,
        'annotations': new_annotations,
        'categories': data['categories']
    }

def main():
    # Paths
    base_dir = "datasets/vizwiz/base_images"
    train_ann = os.path.join(base_dir, "annotations/instances_train.json")
    val_ann = os.path.join(base_dir, "annotations/instances_val.json")
    
    # Output directory
    output_base = "datasets/vizwiz/annotations"
    os.makedirs(output_base, exist_ok=True)
    
    # Load original annotations
    train_data = load_annotations(train_ann)
    val_data = load_annotations(val_ann)
    
    print(f"\nOriginal dataset:")
    print(f"  Train: {len(train_data['images'])} images, {len(train_data['annotations'])} annotations")
    print(f"  Val: {len(val_data['images'])} images, {len(val_data['annotations'])} annotations")
    
    # Process each fold
    for fold_name, (base_ids, novel_ids) in FOLDS.items():
        print(f"\n{'='*70}")
        print(f"Processing {fold_name}")
        print(f"{'='*70}")
        print(f"Base categories: {len(base_ids)}")
        print(f"Novel categories: {len(novel_ids)}")
        
        fold_dir = os.path.join(output_base, fold_name)
        os.makedirs(fold_dir, exist_ok=True)
        
        # Create base splits (no data leakage - exclude images with novel categories)
        print("\nCreating base splits...")
        train_base = filter_annotations_by_categories(train_data, base_ids, keep_images_with_mixed=False)
        val_base = filter_annotations_by_categories(val_data, base_ids, keep_images_with_mixed=False)
        
        print(f"  Train base: {len(train_base['images'])} images, {len(train_base['annotations'])} annotations")
        print(f"  Val base: {len(val_base['images'])} images, {len(val_base['annotations'])} annotations")
        
        # Save base splits
        with open(os.path.join(fold_dir, 'instances_train_base.json'), 'w') as f:
            json.dump(train_base, f)
        with open(os.path.join(fold_dir, 'instances_val_base.json'), 'w') as f:
            json.dump(val_base, f)
        
        # Create novel splits (can include images with base categories for realistic scenarios)
        print("\nCreating novel splits...")
        train_novel = filter_annotations_by_categories(train_data, novel_ids, keep_images_with_mixed=True)
        val_novel = filter_annotations_by_categories(val_data, novel_ids, keep_images_with_mixed=True)
        
        print(f"  Train novel: {len(train_novel['images'])} images, {len(train_novel['annotations'])} annotations")
        print(f"  Val novel: {len(val_novel['images'])} images, {len(val_novel['annotations'])} annotations")
        
        # Save full novel splits
        with open(os.path.join(fold_dir, 'instances_train_novel.json'), 'w') as f:
            json.dump(train_novel, f)
        with open(os.path.join(fold_dir, 'instances_val_novel.json'), 'w') as f:
            json.dump(val_novel, f)
        
        # Create k-shot novel splits
        print("\nCreating k-shot novel splits...")
        for k in [1, 3, 5, 10]:
            train_novel_kshot = create_k_shot_dataset(train_novel, k, seed=0)
            val_novel_kshot = create_k_shot_dataset(val_novel, k, seed=0)
            
            print(f"  {k}-shot train: {len(train_novel_kshot['images'])} images")
            print(f"  {k}-shot val: {len(val_novel_kshot['images'])} images")
            
            with open(os.path.join(fold_dir, f'instances_train_novel_{k}shot.json'), 'w') as f:
                json.dump(train_novel_kshot, f)
            with open(os.path.join(fold_dir, f'instances_val_novel_{k}shot.json'), 'w') as f:
                json.dump(val_novel_kshot, f)
        
        print(f"\n✅ {fold_name} complete! Files saved to {fold_dir}")
    
    print(f"\n{'='*70}")
    print("All folds prepared successfully!")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()

