"""
VizWiz dataset registration for DeFRCN.
Registers all VizWiz few-shot splits following the OD-25ᵢ protocol.
"""

from detectron2.data.datasets import register_coco_instances
import os

ROOT = os.path.expanduser("~/DeFRCN")
DATASET_ROOT = os.path.join(ROOT, "datasets/vizwiz")
IMAGE_ROOT = os.path.join(DATASET_ROOT, "base_images")  # Annotations have "images/train/" prefix

# Skip base datasets if already registered
try:
    register_coco_instances(
        "vizwiz_train",
        {},
        os.path.join(ROOT, "datasets/vizwiz/base_images/annotations/instances_train.json"),
        os.path.join(DATASET_ROOT),
    )
except AssertionError:
    pass  # Already registered

try:
    register_coco_instances(
        "vizwiz_val",
        {},
        os.path.join(ROOT, "datasets/vizwiz/base_images/annotations/instances_val.json"),
        os.path.join(DATASET_ROOT),
    )
except AssertionError:
    pass  # Already registered

# Register fold-specific datasets for few-shot learning
FOLDS = ["OD25_0", "OD25_1", "OD25_2", "OD25_3"]
SPLITS = ["base", "novel"]
K_SHOTS = [1, 3, 5, 10]

for fold in FOLDS:
    fold_ann_dir = os.path.join(DATASET_ROOT, "annotations", fold)
    
    for split in SPLITS:
        # Register train and val for base/novel
        for subset in ["train", "val"]:
            dataset_name = f"vizwiz_{fold}_{subset}_{split}"
            json_file = os.path.join(fold_ann_dir, f"instances_{subset}_{split}.json")
            # Use base_images as root; annotations already have "images/train/" prefix
            img_root = IMAGE_ROOT
            
            register_coco_instances(
                dataset_name,
                {},
                json_file,
                img_root,
            )
        
        # Register k-shot datasets for novel split only
        if split == "novel":
            for k in K_SHOTS:
                for subset in ["train", "val"]:
                    dataset_name = f"vizwiz_{fold}_{subset}_{split}_{k}shot"
                    json_file = os.path.join(fold_ann_dir, f"instances_{subset}_{split}_{k}shot.json")
                    # Use base_images as root; annotations already have "images/train/" prefix
                    img_root = IMAGE_ROOT
                    
                    register_coco_instances(
                        dataset_name,
                        {},
                        json_file,
                        img_root,
                    )

print(f"✅ Registered VizWiz datasets for {len(FOLDS)} folds with k-shot variants")
