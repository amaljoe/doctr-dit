#!/usr/bin/env python3
"""
Python script to run DIT object detection and output results as JSON with bounding boxes.
Similar to dit.sh but outputs raw OCR/detection results in JSON format.
"""

from doctr.utils.visualization import visualize_page
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
import numpy as np

# Add doctr to path for visualization
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

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


def calculate_box_area(bbox: dict) -> float:
    """Calculate the area of a bounding box.

    Args:
        bbox: Dictionary with x1, y1, x2, y2 keys

    Returns:
        Area of the bounding box
    """
    return (bbox["x2"] - bbox["x1"]) * (bbox["y2"] - bbox["y1"])


def calculate_intersection(box1: dict, box2: dict) -> float:
    """Calculate intersection area between two bounding boxes.

    Args:
        box1: First bounding box dictionary with x1, y1, x2, y2 keys
        box2: Second bounding box dictionary with x1, y1, x2, y2 keys

    Returns:
        Intersection area (0 if no intersection)
    """
    # Calculate intersection
    x1_inter = max(box1["x1"], box2["x1"])
    y1_inter = max(box1["y1"], box2["y1"])
    x2_inter = min(box1["x2"], box2["x2"])
    y2_inter = min(box1["y2"], box2["y2"])

    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        return 0.0

    return (x2_inter - x1_inter) * (y2_inter - y1_inter)


def calculate_overlap_ratio(box1: dict, box2: dict) -> tuple[float, float]:
    """Calculate overlap ratio for each box (intersection / box_area).

    Args:
        box1: First bounding box dictionary with x1, y1, x2, y2 keys
        box2: Second bounding box dictionary with x1, y1, x2, y2 keys

    Returns:
        Tuple of (overlap_ratio_box1, overlap_ratio_box2)
        Each ratio is intersection_area / box_area
    """
    intersection = calculate_intersection(box1, box2)

    if intersection == 0.0:
        return (0.0, 0.0)

    area1 = calculate_box_area(box1)
    area2 = calculate_box_area(box2)

    if area1 == 0 or area2 == 0:
        return (0.0, 0.0)

    ratio1 = intersection / area1
    ratio2 = intersection / area2

    return (ratio1, ratio2)


def post_process_detections(detections: list[dict], overlap_threshold: float = 0.75) -> list[dict]:
    """Post-process detections: change title to text and remove overlapping boxes.

    Args:
        detections: List of detection dictionaries
        overlap_threshold: Overlap ratio threshold (0.75 = 75%). If more than this
                         percentage of a box overlaps with another, remove the smaller one.

    Returns:
        Post-processed list of detections
    """
    # Step 1: Change all "title" to "text"
    processed = []
    for detection in detections:
        detection_copy = detection.copy()
        if detection_copy["class"] == "title":
            detection_copy["class"] = "text"
        processed.append(detection_copy)

    # Step 2: Remove overlapping boxes, keeping the larger one
    if len(processed) <= 1:
        return processed

    # Sort by area (largest first) so we prefer larger boxes
    processed.sort(key=lambda d: calculate_box_area(d["bbox"]), reverse=True)

    # Keep only non-overlapping boxes
    final_result = []
    for detection in processed:
        box = detection["bbox"]
        box_area = calculate_box_area(box)
        should_keep = True
        to_remove = []

        # Check against all existing boxes
        for existing in final_result:
            existing_box = existing["bbox"]
            overlap_ratio1, overlap_ratio2 = calculate_overlap_ratio(
                box, existing_box)

            # If more than threshold% of either box is overlapped, remove the smaller one
            if overlap_ratio1 > overlap_threshold or overlap_ratio2 > overlap_threshold:
                existing_area = calculate_box_area(existing_box)
                if box_area < existing_area:
                    # Current box is smaller, skip it
                    should_keep = False
                    break
                else:
                    # Current box is larger or equal, mark existing for removal
                    to_remove.append(existing)

        # Remove boxes that were marked for removal
        for item in to_remove:
            final_result.remove(item)

        # Add current box if it should be kept
        if should_keep:
            final_result.append(detection)

    return final_result


def detections_to_page_format(detections: list[dict], img_height: int, img_width: int) -> dict:
    """Convert detection results to the format expected by visualize_page.

    Args:
        detections: List of detection dictionaries with bbox, class, and confidence
        img_height: Image height in pixels
        img_width: Image width in pixels

    Returns:
        Dictionary in the format expected by visualize_page
    """
    blocks = []

    for detection in detections:
        bbox = detection["bbox"]
        # Convert absolute coordinates to normalized (0-1)
        x1_norm = bbox["x1"] / img_width
        y1_norm = bbox["y1"] / img_height
        x2_norm = bbox["x2"] / img_width
        y2_norm = bbox["y2"] / img_height

        # Create geometry in format ((xmin, ymin), (xmax, ymax))
        geometry = ((x1_norm, y1_norm), (x2_norm, y2_norm))

        # Create a word with the class name as value
        word = {
            "geometry": geometry,
            "value": detection["class"],
            "confidence": detection["confidence"]
        }

        # Create a line containing the word
        line = {
            "geometry": geometry,
            "words": [word]
        }

        # Create a block containing the line
        block = {
            "geometry": geometry,
            "lines": [line],
            "artefacts": []
        }

        blocks.append(block)

    # Create page structure
    page = {
        "dimensions": (img_height, img_width),
        "blocks": blocks
    }

    return page


def visualize_detections(image_path: str, detections: list[dict], output_path: str | None = None,
                         interactive: bool = True, **kwargs) -> None:
    """Visualize detections using doctr's visualize_page function.

    Args:
        image_path: Path to the input image
        detections: List of detection dictionaries
        output_path: Optional path to save the visualization (if None, displays interactively)
        interactive: Whether to show interactive plot
        **kwargs: Additional arguments for visualize_page
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image from {image_path}")

    # Convert BGR to RGB for matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Get image dimensions
    img_height, img_width = img_rgb.shape[:2]

    # Convert detections to page format
    page = detections_to_page_format(detections, img_height, img_width)

    # Visualize
    fig = visualize_page(page, img_rgb, words_only=False,
                         interactive=interactive, **kwargs)

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {output_path}", file=sys.stderr)
    else:
        import matplotlib.pyplot as plt
        plt.show()


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
        "--visualize",
        help="Visualize the detections using matplotlib",
        action="store_true",
    )
    parser.add_argument(
        "--visualize_output",
        help="Path to save visualization image (if not provided, displays interactively)",
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

    # Build detections list
    detections = []
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
        detections.append(detection)

    # Post-process detections: change title to text and remove overlapping boxes
    # If more than 75% of a box overlaps with another, keep only the bigger one
    detections = post_process_detections(detections, overlap_threshold=0.75)

    # Build JSON output with both formats
    results = {
        "image_path": args.image_path,
        "image_size": {
            "width": int(img.shape[1]),
            "height": int(img.shape[0])
        },
        "detections": detections
    }

    # Also include page format for visualize_page compatibility
    page_format = detections_to_page_format(
        detections, int(img.shape[0]), int(img.shape[1]))
    results["page_format"] = page_format

    # Output JSON
    json_output = json.dumps(results, indent=2)

    if args.output_file:
        with open(args.output_file, 'w') as f:
            f.write(json_output)
        print(f"Results saved to {args.output_file}", file=sys.stderr)
    else:
        print(json_output)

    # Visualize if requested
    if args.visualize:
        visualize_detections(
            args.image_path,
            detections,
            output_path=args.visualize_output,
            interactive=(args.visualize_output is None)
        )


if __name__ == '__main__':
    main()
