AI TOOTH SEGMENTATION

====================================

venv\Scripts\activate

====================================

yolo detect train data=data.yaml model=yolov8n.pt imgsz=1024 epochs=100 batch=4

====================================

yolo detect predict model=runs/detect/train/weights/best.pt source=dataset/test/1.jpg conf=0.25 save=True