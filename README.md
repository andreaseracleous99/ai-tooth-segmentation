AI TOOTH SEGMENTATION

====================================

venv\Scripts\activate

====================================

yolo detect train model=yolo11n.pt data=datasets/train-yolo/merged/data.yaml imgsz=1024 epochs=100 batch=8

====================================

yolo detect predict model=runs/detect/train/weights/best.pt source=path\to\radiograph.jpg save=True conf=0.25