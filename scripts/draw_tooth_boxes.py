from ultralytics import YOLO
from pathlib import Path

MODEL = "../models/tooth_boxes_v2.pt"
IMG = "../datasets/tooth_boxes/11/images/100.jpg"

CONF = 0.25 
IMGSZ = 1024 

PROJECT = "../runs/detect"
NAME = "predict" 

def main():
    img_path = Path(IMG)
    if not img_path.exists():
        raise FileNotFoundError(img_path.resolve())

    model = YOLO(MODEL)

    results = model.predict(
        source=IMG,
        conf=CONF,
        imgsz=IMGSZ,
        save=True,
        hide_labels=True,
        hide_conf=True,
        project=PROJECT,
        name=NAME,
        exist_ok=True, 
        verbose=False,
    )

    print("Saved to:", results[0].save_dir)

if __name__ == "__main__":
    main()
