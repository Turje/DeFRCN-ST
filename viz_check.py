import json, os, random, cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = "datasets/vizwiz/base_images"
TRAIN_JSON = f"{ROOT}/annotations/instances_train.json"
VAL_JSON   = f"{ROOT}/annotations/instances_val.json"
os.makedirs("vizcheck", exist_ok=True)

def load_coco(ann_path):
    with open(ann_path, "r") as f:
        data = json.load(f)
    cats = {c["id"]: c["name"] for c in data["categories"]}
    imgs = {im["id"]: im for im in data["images"]}
    by_img = {}
    for a in data["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)
    return imgs, by_img, cats

def resolve_img_path(file_name, split):
    # 1) try as-is (it might already include 'images/train/' or 'images/val/')
    p1 = os.path.join(ROOT, file_name)
    if os.path.exists(p1): return p1
    # 2) force the split folder explicitly
    p2 = os.path.join(ROOT, "images", split, os.path.basename(file_name))
    return p2

def show_one(ann_path, split, out_png, title_prefix):
    imgs, by_img, cats = load_coco(ann_path)
    cands = [iid for iid, anns in by_img.items() if anns]
    random.seed(1337)
    img_id = random.choice(cands)
    meta = imgs[img_id]
    img_path = resolve_img_path(meta["file_name"], split)
    assert os.path.exists(img_path), f"Image not found: {img_path}"

    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(8,8))
    ax.imshow(img); ax.axis('off')
    for a in by_img[img_id]:
        x,y,w,h = a["bbox"]
        rect = Rectangle((x,y), w,h, fill=False, linewidth=2)
        ax.add_patch(rect)
        ax.text(x, max(0,y-2), cats.get(a["category_id"], str(a["category_id"])),
                fontsize=10, bbox=dict(facecolor='yellow', alpha=0.6, pad=1.5))
    ax.set_title(f"{title_prefix}: {os.path.basename(img_path)}")
    fig.savefig(out_png, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"✅ saved {out_png}")

show_one(TRAIN_JSON, split="train", out_png="vizcheck/train_sample.png", title_prefix="TRAIN sample")
show_one(VAL_JSON,   split="val",   out_png="vizcheck/val_sample.png",   title_prefix="VAL sample")
