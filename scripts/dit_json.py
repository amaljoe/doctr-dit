#!/usr/bin/env python3
"""
Python script to run DIT object detection and output results as JSON with bounding boxes.
Similar to dit.sh but outputs raw OCR/detection results in JSON format.
"""

from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.config import get_cfg
import torch
from ditod import add_vit_config
import torch.functional
import torch.utils.checkpoint
import argparse
import ssl
import warnings
import os
import json
import sys

import cv2

# Add dit/object_detection to path for ditod import
sys.path.insert(0, os.path.join(os.path.dirname(
    __file__), '..', 'dit', 'object_detection'))


# Suppress only necessary warnings (keep SSL and torch warnings suppressed)
warnings.filterwarnings("ignore")
os.environ['PYTHONWARNINGS'] = 'ignore'


# Suppress torch warnings by monkey patching
original_checkpoint = torch.utils.checkpoint.checkpoint


def silent_checkpoint(*args, **kwargs):
    kwargs.setdefault('use_reentrant', False)
    return original_checkpoint(*args, **kwargs)


torch.utils.checkpoint.checkpoint = silent_checkpoint

# Suppress torch functional warnings
original_meshgrid = torch.functional.meshgrid


def silent_meshgrid(*args, **kwargs):
    kwargs.setdefault('indexing', 'ij')
    return original_meshgrid(*args, **kwargs)


torch.functional.meshgrid = silent_meshgrid

# Fix SSL certificate verification issues
ssl._create_default_https_context = ssl._create_unverified_context


def main():
    parser = argparse.ArgumentParser(
        description="Detectron2 inference script - JSON output")
    parser.add_argument(
        "--image_path",
        help="Path to input image",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_file",
        help="Path to output JSON file (default: stdout)",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--config-file",
        default="dit/object_detection/publaynet_configs/maskrcnn/maskrcnn_dit_large.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs='*',
    )

    args = parser.parse_args()

    # Step 1: instantiate config
    cfg = get_cfg()
    add_vit_config(cfg)
    cfg.merge_from_file(args.config_file)

    # Step 2: add model weights URL to config
    # Filter out empty strings from opts
    opts = [opt for opt in args.opts if opt] if args.opts else []
    if opts:
        cfg.merge_from_list(opts)

    # Step 3: set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.MODEL.DEVICE = device

    # Step 4: define model
    predictor = DefaultPredictor(cfg)

    # Step 5: run inference
    img = cv2.imread(args.image_path)
    if img is None:
        raise ValueError(f"Could not read image from {args.image_path}")

    # Get metadata for class names
    md = MetadataCatalog.get(cfg.DATASETS.TEST[0])
    if cfg.DATASETS.TEST[0] == 'icdar2019_test':
        thing_classes = ["table"]
    else:
        thing_classes = ["text", "title", "list", "table", "figure"]
    md.set(thing_classes=thing_classes)

    # Run prediction
    output = predictor(img)["instances"]

    # Extract bounding boxes, scores, and classes
    # Shape: (N, 4) as [x1, y1, x2, y2]
    boxes = output.pred_boxes.tensor.cpu().numpy()
    scores = output.scores.cpu().numpy()  # Shape: (N,)
    classes = output.pred_classes.cpu().numpy()  # Shape: (N,)

    # Build JSON output
    results = {
        "image_path": args.image_path,
        "image_size": {
            "width": int(img.shape[1]),
            "height": int(img.shape[0])
        },
        "detections": []
    }

    for i in range(len(boxes)):
        box = boxes[i]
        detection = {
            "bbox": {
                "x1": float(box[0]),
                "y1": float(box[1]),
                "x2": float(box[2]),
                "y2": float(box[3]),
                "width": float(box[2] - box[0]),
                "height": float(box[3] - box[1])
            },
            "class": thing_classes[int(classes[i])],
            "confidence": float(scores[i])
        }
        results["detections"].append(detection)

    # Output JSON
    json_output = json.dumps(results, indent=2)

    if args.output_file:
        with open(args.output_file, 'w') as f:
            f.write(json_output)
        print(f"Results saved to {args.output_file}", file=sys.stderr)
    else:
        print(json_output)


if __name__ == '__main__':
    main()
