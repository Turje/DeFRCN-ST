"""
VizWiz Few-Shot OD-25ᵢ Fold Definitions for DeFRCN

This module defines the four folds for the VizWiz Few-Shot Object Detection protocol
following the OD-25ᵢ protocol.

Each fold contains:
- 75 base classes (for base training)
- 25 novel classes (for few-shot fine-tuning)
"""

# All 100 VizWiz categories (ID: name mapping)
ALL_CATEGORIES = {
    1: "album", 2: "apple", 3: "backpack", 4: "banana", 5: "bar", 6: "bed", 7: "bird", 
    8: "book", 9: "bottle", 10: "bowl", 11: "bracelet", 12: "broccoli", 13: "cake", 
    14: "calculator", 15: "car", 16: "carrot", 17: "cash", 18: "cat", 19: "ceiling_fan", 
    20: "cell_phone", 21: "cereal_box", 22: "chair", 23: "clock", 24: "coin", 
    25: "computer_keyboard", 26: "computer_mouse", 27: "couch", 28: "crockpot", 29: "cup", 
    30: "curtain", 31: "dial", 32: "dog", 33: "dog_collar", 34: "drawer", 35: "electric_fan", 
    36: "envelope", 37: "flashdrive", 38: "food_menu", 39: "fork", 40: "gift_card", 
    41: "guitar", 42: "hat", 43: "house", 44: "ipad", 45: "key", 46: "knife", 47: "lamp", 
    48: "landline_phone", 49: "laptop", 50: "laundry_machine", 51: "magazine", 52: "microphone", 
    53: "microwave", 54: "monitor", 55: "newspaper", 56: "orange", 57: "oven", 58: "packet", 
    59: "painting", 60: "pen", 61: "perfume", 62: "person", 63: "piano", 64: "pillow", 
    65: "pizza", 66: "plate", 67: "printer", 68: "purse", 69: "ramen", 70: "receipt", 
    71: "refrigerator", 72: "remote", 73: "ring", 74: "rug", 75: "sandal", 76: "sandwich", 
    77: "scale", 78: "shoe", 79: "sign", 80: "sink", 81: "sock", 82: "speaker", 83: "spoon", 
    84: "stapler", 85: "sticker", 86: "stool", 87: "stove", 88: "strawberry", 89: "suitcase", 
    90: "sweatshirt", 91: "television", 92: "toaster", 93: "towel", 94: "truck", 95: "tube", 
    96: "vacuum", 97: "vase", 98: "wallet", 99: "watch", 100: "wine"
}

# Novel classes for each fold (25 novel categories per fold)
FOLD_NOVEL_CLASSES = {
    "OD25_0": [
        "couch", "watch", "drawer", "landline_phone", "strawberry", "painting", "pillow",
        "envelope", "bracelet", "calculator", "car", "cash", "cat", "coin", "crockpot",
        "guitar", "fork", "monitor", "pen", "ring", "purse", "sink", "sock", 
        "television", "truck"
    ],
    "OD25_1": [
        "bird", "album", "ceiling_fan", "cereal_box", "dial", "dog_collar", "electric_fan",
        "food_menu", "gift_card", "ipad", "microphone", "piano", "printer", "sandal",
        "scale", "shoe", "sign", "stapler", "sticker", "stool", "stove", "tube",
        "towel", "vase", "wallet"
    ],
    "OD25_2": [
        "backpack", "banana", "bowl", "broccoli", "chair", "clock", "computer_mouse",
        "curtain", "flashdrive", "hat", "house", "key", "knife", "laundry_machine",
        "magazine", "newspaper", "orange", "oven", "packet", "perfume", "pizza",
        "ramen", "receipt", "suitcase", "sweatshirt"
    ],
    "OD25_3": [
        "apple", "bar", "bed", "book", "bottle", "cake", "cell_phone", "computer_keyboard",
        "cup", "dog", "lamp", "laptop", "microwave", "person", "plate", "refrigerator",
        "remote", "rug", "sandwich", "speaker", "spoon", "toaster", "towel", 
        "vacuum", "wine"
    ]
}

def get_category_id_by_name(category_name):
    """Get category ID by name."""
    for cid, cname in ALL_CATEGORIES.items():
        if cname == category_name:
            return cid
    return None

def get_base_novel_ids(fold_name):
    """
    Get base and novel category IDs for a specific fold.
    
    Returns:
        base_ids (list): List of 75 base category IDs
        novel_ids (list): List of 25 novel category IDs
    """
    if fold_name not in FOLD_NOVEL_CLASSES:
        raise ValueError(f"Unknown fold: {fold_name}")
    
    # Get novel category IDs
    novel_ids = [get_category_id_by_name(name) for name in FOLD_NOVEL_CLASSES[fold_name]]
    
    # Get base category IDs (all categories minus novel)
    all_ids = set(ALL_CATEGORIES.keys())
    novel_ids_set = set(novel_ids)
    base_ids = sorted(list(all_ids - novel_ids_set))
    
    return base_ids, sorted(novel_ids)

# Export fold definitions
FOLDS = {
    "OD25_0": get_base_novel_ids("OD25_0"),
    "OD25_1": get_base_novel_ids("OD25_1"),
    "OD25_2": get_base_novel_ids("OD25_2"),
    "OD25_3": get_base_novel_ids("OD25_3"),
}

if __name__ == "__main__":
    # Print fold statistics
    for fold_name, (base_ids, novel_ids) in FOLDS.items():
        print(f"\n{fold_name}:")
        print(f"  Base categories: {len(base_ids)} (IDs: {base_ids[:5]}...{base_ids[-3:]})")
        print(f"  Novel categories: {len(novel_ids)} (IDs: {novel_ids[:5]}...{novel_ids[-3:]})")

