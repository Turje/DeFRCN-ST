import json, os, random, argparse, collections

def main(args):
    random.seed(0)  # fixed seed for exact reproducibility
    base_json = args.base_json
    folds_json = args.folds_json
    out_root  = args.out_root
    ks = [int(k) for k in args.k.split(",")]

    data = json.load(open(base_json))
    images = {im["id"]: im for im in data["images"]}
    anns   = data["annotations"]
    cats   = {c["id"]: c for c in data["categories"]}

    # map: cat name -> id (for your official_folds names)
    name2id = {c["name"]: cid for cid, c in ((c["id"], c) for c in data["categories"])}

    folds = json.load(open(folds_json))
    for fi in range(4):
        fold_key = f"fold{fi}"
        novel_names = folds[fold_key]
        novel_ids   = [name2id[n] for n in novel_names]

        # collect anns per novel cat (stable order: by (image_id, annotation id))
        per_cat = collections.defaultdict(list)
        for a in anns:
            if a["category_id"] in novel_ids:
                per_cat[a["category_id"]].append(a)
        for cid in per_cat:
            per_cat[cid].sort(key=lambda a: (a["image_id"], a["id"]))

        # for each K produce a JSON
        for K in ks:
            sel_anns = []
            sel_img_ids = set()
            for cid in novel_ids:
                pool = per_cat.get(cid, [])
                if len(pool) < K:
                    raise ValueError(f"Fold {fi}: category {cats[cid]['name']} has only {len(pool)} anns, need {K}.")
                chosen = pool[:K]  # deterministic: first K after stable sort
                sel_anns.extend(chosen)
                sel_img_ids.update(a["image_id"] for a in chosen)

            # build minimal COCO
            out_images = [images[i] for i in sorted(sel_img_ids)]
            out_cats   = [cats[cid] for cid in sorted(novel_ids)]
            out = {
                "images": out_images,
                "annotations": [
                    {k:v for k,v in a.items()} for a in sel_anns
                ],
                "categories": out_cats,
            }

            out_dir = os.path.join(out_root, f"fold{fi}")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"full_box_{K}shot_all_trainval.json")
            with open(out_path, "w") as f:
                json.dump(out, f)
            print(f"✅ wrote: {out_path} | imgs={len(out_images)} anns={len(sel_anns)} cats={len(out_cats)}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-json", required=True)
    ap.add_argument("--folds-json", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--k", default="1,3,5,10")
    args = ap.parse_args()
    main(args)
