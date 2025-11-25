import os
import json
from glob import glob
from PIL import Image

def load_report_with_images(
    json_dir="../data/DITrans-EMNLP/political_report/jsons/",
    img_dir ="../data/DITrans-EMNLP/political_report/imgs/",
    limit=None        # <── here
):
    json_paths = sorted(glob(os.path.join(json_dir, "*.json")))

    if limit is not None:
        json_paths = json_paths[:limit]

    items = []

    for jp in json_paths:
        base = os.path.splitext(os.path.basename(jp))[0]

        # match image
        img_path = None
        for ext in ["jpg", "jpeg", "png"]:
            candidate = os.path.join(img_dir, f"{base}.{ext}")
            if os.path.exists(candidate):
                img_path = candidate
                break

        if img_path is None:
            print(f"[WARN] No image for {base}")
            continue

        with open(jp, "r", encoding="utf-8") as f:
            data = json.load(f)

        img = Image.open(img_path).convert("RGB")

        items.append((base, data, img, img_path))


    return items

def boxes_overlap(b1, b2, threshold=0.01):
    """Return True if IoU > threshold."""
    xa = max(b1["x_min"], b2["x_min"])
    ya = max(b1["y_min"], b2["y_min"])
    xb = min(b1["x_max"], b2["x_max"])
    yb = min(b1["y_max"], b2["y_max"])

    inter_w = max(0, xb - xa)
    inter_h = max(0, yb - ya)
    inter = inter_w * inter_h

    area1 = (b1["x_max"] - b1["x_min"]) * (b1["y_max"] - b1["y_min"])
    area2 = (b2["x_max"] - b2["x_min"]) * (b2["y_max"] - b2["y_min"])

    union = area1 + area2 - inter
    if union == 0:
        return False

    return inter / union > threshold

def box_contained(small, big, threshold=0.9):
    """
    Check if the small box is at least `threshold` (e.g., 0.9 = 90%)
    inside the big box.

    small, big: dicts with keys x_min, y_min, x_max, y_max
    """
    # Intersection coordinates
    xa = max(small["x_min"], big["x_min"])
    ya = max(small["y_min"], big["y_min"])
    xb = min(small["x_max"], big["x_max"])
    yb = min(small["y_max"], big["y_max"])

    inter_w = max(0, xb - xa)
    inter_h = max(0, yb - ya)
    inter_area = inter_w * inter_h

    # Small box area
    small_area = (small["x_max"] - small["x_min"]) * (small["y_max"] - small["y_min"])
    
    if small_area == 0 or inter_area == 0:
        return False

    # Fraction of the small box that lies inside the big box
    inside_ratio = inter_area / small_area

    return inside_ratio >= threshold



def combine_overlapping(sentences):
    """Merge overlapping sentence boxes (keep upper-left one)."""
    ids = sorted(sentences.keys())
    merged = {}
    used = set()

    for i in ids:
        if i in used:
            continue

        s1 = sentences[i]
        box1 = s1["box"]

        # this cluster will merge all overlaps into s1
        cluster_ids = [i]

        for j in ids:
            if j in cluster_ids or j in used:
                continue
            s2 = sentences[j]
            box2 = s2["box"]

            if boxes_overlap(box1, box2):
                # add to cluster
                cluster_ids.append(j)

                # merge bounding box
                box1 = {
                    "x_min": min(box1["x_min"], box2["x_min"]),
                    "y_min": min(box1["y_min"], box2["y_min"]),
                    "x_max": max(box1["x_max"], box2["x_max"]),
                    "y_max": max(box1["y_max"], box2["y_max"]),
                }

        # Determine cluster head (top-left)
        top_left = min(
            cluster_ids,
            key=lambda k: (sentences[k]["box"]["y_min"], sentences[k]["box"]["x_min"])
        )

        # Merge texts
        merged_text = ""
        merged_tgt = ""

        for cid in sorted(cluster_ids):
            merged_text += " " + sentences[cid]["src"]
            if sentences[cid]["tgt"]:
                merged_tgt += " " + sentences[cid]["tgt"]

        merged[top_left] = {
            "src": merged_text.strip(),
            "tgt": merged_tgt.strip(),
            "sen_id": top_left,
            "box": box1,
        }

        used.update(cluster_ids)

    return merged

def get_docs(items):
    docs = []
    for filename, data, image, _ in items:
        sentences = {}
        for obj in data.get("objects", []):
            for sen in obj.get("sentences", []):
                info = sen.get("line_info", {})
                sen_id = sen.get("sen_id")
                src = info.get("text_gt")
                tgt = info.get("text_trans_ref")
                box = info.get("line_box")
                x_min, y_min, x_max, y_max = box[0]['x'], box[0]['y'], box[2]['x'], box[2]['y']
                if sen_id in sentences.keys():
                    if src[0] != ' ':
                        src = ' ' + src
                    sentences[sen_id]['src'] += src
                    curr_box = sentences[sen_id]['box']
                    sentences[sen_id]['box'] = {
                        'x_min': min(x_min, curr_box['x_min']),
                        'y_min': min(y_min, curr_box['y_min']),
                        'x_max': max(x_max, curr_box['x_max']),
                        'y_max': max(y_max, curr_box['y_max']),
                    }
                else:
                    sentences[sen_id] = {
                        'src': src,
                        'tgt': tgt,
                        'sen_id': sen_id,
                        'box': {
                            'x_min': x_min,
                            'y_min': y_min,
                            'x_max': x_max,
                            'y_max': y_max,
                        }
                    }
            sentences = combine_overlapping(sentences)
    
        docs.append({
            "sentences": sentences,
        })
    return docs


if __name__ == '__main__':
    items = load_report_with_images(limit=10)
    print("Loaded:", len(items))
    docs = get_docs(items)
    