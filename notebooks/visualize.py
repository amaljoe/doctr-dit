import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

def visualize_word_boxes(image_path, boxes, labels, alpha=120):
    """
    Draw word bounding boxes on the image.
    Each bounding box is colored according to its segment label.

    image_path : path to image
    boxes      : list of [x_min, y_min, x_max, y_max]
    labels     : list of integers (segment_id for each word)
    """

    # Distinct colors for up to 10 segments (expand if needed)
    COLORS = [
        "red", "blue", "green", "orange", "purple",
        "cyan", "magenta", "yellow", "lime", "pink"
    ]

    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")

    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = map(float, box)

        color = COLORS[label % len(COLORS)]
        
        # Semi-transparent fill
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
    
    return img