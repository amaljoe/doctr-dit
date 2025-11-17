import cv2
from doclayout_yolo import YOLOv10
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import pipeline
import logging


def detect_text_yolo(page):
    """Detect text regions in an image using YOLOv10 model.
    Args:
        page: Input image as a NumPy array.
    Returns:
        det_res: Detection results from the YOLOv10 model.
    """
    # Load the pre-trained model
    model = YOLOv10("models/yolo.pt")

    # Perform prediction
    det_res = model.predict(
        page,   # Image to predict
        imgsz=1024,        # Prediction image size
        conf=0.2,          # Confidence threshold
        device="mps"
    )

    return det_res


def box_area(box):
    # box = [x1, y1, x2, y2]
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def intersection_area(word_box, big_box):
    x1 = max(word_box[0][0], big_box[0])
    y1 = max(word_box[0][1], big_box[1])
    x2 = min(word_box[1][0], big_box[2])
    y2 = min(word_box[1][1], big_box[3])

    if x2 <= x1 or y2 <= y1:
        return 0
    return (x2 - x1) * (y2 - y1)


def get_big_boxes(page_export, det_res):
    big_boxes = det_res[0].boxes.xyxyn.tolist()
    big_boxes_children = [[] for _ in big_boxes]

    for idx, line in enumerate(page_export['blocks'][0]['lines']):
        for word in line['words']:
            word_box = word['geometry']
            word_area = box_area(
                [word_box[0][0], word_box[0][1], word_box[1][0], word_box[1][1]])

            candidates = []

            # Find all boxes with >= 75% overlap
            for i, box in enumerate(big_boxes):
                inter_area = intersection_area(word_box, box)
                overlap_ratio = inter_area / (word_area + 1e-8)

                if overlap_ratio >= 0.75:
                    candidates.append((i, overlap_ratio, box_area(box)))

            if candidates:
                # Sort by:
                #   1) highest overlap
                #   2) smallest big-box area (closest fit)
                candidates.sort(key=lambda x: (x[2]))

                best_match = candidates[0][0]

                word['line_id'] = idx
                big_boxes_children[best_match].append(word)
            else:
                print("No 75% match:", word)
    return big_boxes, big_boxes_children


def get_line(box, geometry):
    text = ""
    for child in box:
        text += child['value'] + " "
    geometry = ((np.float32(geometry[0]), np.float32(
        geometry[1])), (np.float32(geometry[2]), np.float32(geometry[3])))
    num_lines = len(set([child['line_id'] for child in box]))
    return {
        "geometry": geometry,
        "num_lines": num_lines,
        "words": [
            {
                "confidence": 1,
                "crop_orientation": {
                    'value': 0,
                    'confidence':  None
                },
                "value": text.strip(),
                "geometry": geometry
            }]
    }


def update_page_with_layout(page, page_export):
    det_res = detect_text_yolo(page)
    big_boxes, big_boxes_children = get_big_boxes(page_export, det_res)
    types = det_res[0].boxes.data[:, -1].int().cpu()
    mask = (0, 1, 4, 6, 7, 9)
    types = torch.isin(types, torch.tensor(mask))
    lines = [get_line(big_boxes_children[i], big_boxes[i]) for i in range(
        len(big_boxes)) if big_boxes_children[i] and types[i]]
    page_export['blocks'][0]['lines'] = lines
    return page_export


def translate_lines(page_export):
    translator = pipeline("translation", model="facebook/nllb-200-distilled-600M", src_lang="eng_Latn", tgt_lang="mal_Mlym")
    values = []
    for block in page_export['blocks']:
        for line in block['lines']:
            if len(line['words']) > 1:
                logging.warning("More than 1 word in line")
            for word in line['words']:
                values.append(word['value'])
    translations = translator(values)
    for block in page_export['blocks']:
        for line in block['lines']:
            for word in line['words']:
                word['value'] = translations.pop(0)['translation_text']
    return page_export


if __name__ == "__main__":
    from backend.pytorch import DET_ARCHS, RECO_ARCHS, forward_image, load_predictor
    from doctr.io import DocumentFile
    import torch
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt
    import numpy as np
    import streamlit as st
    import torch
    from backend.pytorch import DET_ARCHS, RECO_ARCHS, forward_image, load_predictor
    from doctr.utils.visualization import visualize_page

    doc_path = "data/mydata/flight.png"
    if doc_path.endswith((".pdf", ".PDF")):
        doc = DocumentFile.from_pdf(doc_path)
    else:
        doc = DocumentFile.from_images(doc_path)
    page = doc[0]

    forward_device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu")

    predictor = load_predictor(
        det_arch=DET_ARCHS[0],
        reco_arch=RECO_ARCHS[0],
        assume_straight_pages=True,
        straighten_pages=False,
        export_as_straight_boxes=False,
        disable_page_orientation=False,
        disable_crop_orientation=False,
        bin_thresh=0.3,
        box_thresh=0.1,
        device=forward_device,
    )

    # Forward the image to the model
    seg_map = forward_image(predictor, page, forward_device)
    seg_map = np.squeeze(seg_map)
    seg_map = cv2.resize(
        seg_map, (page.shape[1], page.shape[0]), interpolation=cv2.INTER_LINEAR)

    # Run OCR
    out = predictor([page])
    page_export = out.pages[0].export()
    img = out.pages[0].synthesize()

    page_export = update_page_with_layout(page, page_export)
