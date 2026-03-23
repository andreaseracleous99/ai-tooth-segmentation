AI TOOTH SEGMENTATION

====================================

## Project Structure

- `app.py`: Main Streamlit application for UI
- `scripts/`: Python scripts for training, prediction, and data processing
- `datasets/`: Dataset directories
- `models/`: Trained model files
- `outputs/`: Output files and results
- `runs/`: YOLO training runs

## Setup

venv\Scripts\activate

## Usage

- Run app: `streamlit run app.py`
- Train models: See scripts/ directory
- Evaluate: `python scripts/evaluate_binary.py`

====================================

dataset 30 -> ekana replace olous tous numbers se 0

====================================

yolo detect predict model=models/tooth_boxes_v2.pt source=datasets/tooth_vs_nontooth/20/images/1.png save=True conf=0.25
