AI TOOTH SEGMENTATION

====================================

venv\Scripts\activate

====================================

dataset 30 -> ekana replace olous tous numbers se 0

====================================

yolo detect predict model=models/tooth_boxes_v2.pt source=datasets/tooth_vs_nontooth/20/images/1.png save=True conf=0.25
