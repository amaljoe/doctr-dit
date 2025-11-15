import cv2
from doclayout_yolo import YOLOv10

# Load the pre-trained model
model = YOLOv10("models/yolo.pt")

# Perform prediction
det_res = model.predict(
    "data/mydata/flight.png",   # Image to predict
    imgsz=1024,        # Prediction image size
    conf=0.2,          # Confidence threshold
    device="mps"
)

# Annotate and save the result
annotated_frame = det_res[0].plot(pil=True, line_width=5, font_size=20)
cv2.imwrite("outputs/mydata/flight.jpg", annotated_frame)
