"""
Multiclass Tooth Detection and Classification Pipeline

Complete pipeline for:
1. Validating intraoral/radiograph image
2. Detecting tooth regions using YOLO
3. Classifying each tooth into 32 FDI classes
4. Determining tooth groups (incisor, canine, premolar, molar)
5. Providing comprehensive tooth segmentation results
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np
from ultralytics import YOLO

from tooth_system import get_tooth_info, get_all_fdi_numbers, ToothGroup


class ToothDetectionResult:
    """Container for tooth detection results"""
    def __init__(self, 
                 tooth_id: int,
                 fdi_number: int,
                 confidence: float,
                 bbox: Tuple[int, int, int, int],
                 group: ToothGroup,
                 tooth_patch: Optional[Image.Image] = None):
        self.tooth_id = tooth_id
        self.fdi_number = fdi_number
        self.confidence = confidence
        self.bbox = bbox  # (x, y, w, h)
        self.group = group
        self.tooth_patch = tooth_patch
        self.tooth_info = get_tooth_info(fdi_number)

    def to_dict(self) -> Dict:
        """Convert result to dictionary"""
        return {
            "tooth_id": self.tooth_id,
            "fdi_number": self.fdi_number,
            "fdi_name": self.tooth_info["name"],
            "confidence": float(self.confidence),
            "bbox": self.bbox,
            "group": self.group.value,
            "arch": self.tooth_info["arch"].value,
            "side": self.tooth_info["side"],
        }

    def __repr__(self):
        return f"Tooth(FDI={self.fdi_number}, Conf={self.confidence:.3f}, Group={self.group.value})"


class MulticlassToothClassifier:
    """Multiclass classifier for 32 individual teeth"""

    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model_path = Path(model_path)
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Load model
        ckpt = torch.load(self.model_path, map_location=self.device, weights_only=True)
        
        self.classes = ckpt["classes"]
        self.img_size = ckpt.get("img_size", 224)
        self.mean = ckpt.get("mean", (0.485, 0.456, 0.406))
        self.std = ckpt.get("std", (0.229, 0.224, 0.225))
        self.num_classes = ckpt.get("num_classes", len(self.classes))
        self.arch = ckpt.get("arch", "resnet18")

        # Build model
        if self.arch == "resnet18":
            self.model = models.resnet18(weights=None)
            self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)
        elif self.arch == "resnet50":
            self.model = models.resnet50(weights=None)
            self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)
        elif self.arch == "efficientnet_v2_s":
            self.model = models.efficientnet_v2_s(weights=None)
            in_features = self.model.classifier[-1].in_features
            self.model.classifier[-1] = nn.Linear(in_features, self.num_classes)
        else:
            raise ValueError(f"Unsupported checkpoint architecture: {self.arch}")
        self.model.load_state_dict(ckpt["model_state"])
        self.model.to(self.device).eval()

        # Preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ])

    @torch.no_grad()
    def predict(self, image: Image.Image) -> Tuple[int, float, Dict]:
        """
        Predict tooth class for an image patch
        
        Returns:
            fdi_number: FDI tooth number (11-48)
            confidence: Prediction confidence (0-1)
            probs_dict: Dictionary of all class probabilities
        """
        x = self.transform(image).unsqueeze(0).to(self.device)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu()
        
        pred_idx = int(probs.argmax().item())
        pred_class = int(self.classes[pred_idx])
        confidence = float(probs[pred_idx].item())
        
        probs_dict = {int(self.classes[i]): float(p) for i, p in enumerate(probs.tolist())}
        
        return pred_class, confidence, probs_dict


class ToothDetectionPipeline:
    """Complete pipeline for tooth detection and classification"""

    def __init__(self,
                 radiograph_model_path: str,
                 yolo_model_path: str,
                 multiclass_model_path: str,
                 device: str = "cuda"):
        """
        Initialize pipeline with all required models
        
        Args:
            radiograph_model_path: Path to radiograph classifier
            yolo_model_path: Path to YOLO detection model
            multiclass_model_path: Path to multiclass tooth classifier
            device: "cuda" or "cpu"
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        
        # Load radiograph classifier
        self.radiograph_classifier = self._load_radiograph_classifier(radiograph_model_path)
        
        # Load YOLO detector
        self.yolo_model = YOLO(yolo_model_path)
        
        # Load multiclass classifier
        self.tooth_classifier = MulticlassToothClassifier(multiclass_model_path, self.device)

    def _load_radiograph_classifier(self, model_path: str):
        """Load radiograph classifier model"""
        model_path = Path(model_path)
        if not model_path.exists():
            print(f"[WARN] Radiograph model not found: {model_path}")
            return None
        
        ckpt = torch.load(model_path, map_location=self.device, weights_only=True)
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(ckpt["classes"]))
        model.load_state_dict(ckpt["model_state"])
        model.to(self.device).eval()
        
        return {
            "model": model,
            "classes": ckpt["classes"],
            "img_size": ckpt.get("img_size", 224),
            "mean": ckpt.get("mean", (0.485, 0.456, 0.406)),
            "std": ckpt.get("std", (0.229, 0.224, 0.225)),
        }

    def validate_radiograph(self, image: Image.Image, confidence_threshold: float = 0.5) -> Tuple[bool, float]:
        """
        Validate if image is an intraoral radiograph
        
        Returns:
            is_valid: True if image is a valid radiograph
            confidence: Confidence score
        """
        if self.radiograph_classifier is None:
            return True, 1.0  # Skip if model not available
        
        model = self.radiograph_classifier["model"]
        classes = self.radiograph_classifier["classes"]
        img_size = self.radiograph_classifier["img_size"]
        mean = self.radiograph_classifier["mean"]
        std = self.radiograph_classifier["std"]
        
        tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        
        with torch.no_grad():
            x = tf(image).unsqueeze(0).to(self.device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0].cpu()
            pred_idx = int(probs.argmax().item())
            pred_class = classes[pred_idx]
            confidence = float(probs[pred_idx].item())
        
        is_radiograph = "radiograph" in pred_class.lower()
        return is_radiograph, confidence

    def detect_and_classify(self, 
                           image: Image.Image,
                           conf_threshold: float = 0.25,
                           imgsz: int = 1024) -> Dict:
        """
        Complete detection and classification pipeline
        
        Returns:
            Dictionary containing:
            - is_radiograph: bool
            - radiograph_confidence: float
            - detections: List[ToothDetectionResult]
            - teeth_by_group: Dict mapping ToothGroup to list of detected teeth
        """
        result_dict = {
            "is_radiograph": None,
            "radiograph_confidence": None,
            "detections": [],
            "teeth_by_group": {
                ToothGroup.INCISOR: [],
                ToothGroup.CANINE: [],
                ToothGroup.PREMOLAR: [],
                ToothGroup.MOLAR: [],
            },
        }

        # Step 1: Validate radiograph
        is_radio, radio_conf = self.validate_radiograph(image, conf_threshold)
        result_dict["is_radiograph"] = is_radio
        result_dict["radiograph_confidence"] = radio_conf
        
        if not is_radio:
            print(f"[WARN] Image may not be an intraoral radiograph (confidence: {radio_conf:.3f})")

        # Step 2: Detect teeth using YOLO
        results = self.yolo_model.predict(
            source=image,
            conf=conf_threshold,
            imgsz=imgsz,
            save=False,
            verbose=False,
        )

        if results is None or len(results) == 0:
            return result_dict

        result = results[0]
        
        if result.boxes is None or len(result.boxes) == 0:
            return result_dict

        # Step 3: Classify each detected tooth
        tooth_id = 0
        for box_idx, box in enumerate(result.boxes):
            # Extract bbox
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
            
            # Extract tooth patch
            tooth_patch = image.crop((int(x1), int(y1), int(x2), int(y2)))
            
            # Classify tooth
            try:
                fdi_number, confidence, probs = self.tooth_classifier.predict(tooth_patch)
                tooth_info = get_tooth_info(fdi_number)
                group = tooth_info["group"]
                
                detection = ToothDetectionResult(
                    tooth_id=tooth_id,
                    fdi_number=fdi_number,
                    confidence=confidence,
                    bbox=(x, y, w, h),
                    group=group,
                    tooth_patch=tooth_patch,
                )
                
                result_dict["detections"].append(detection)
                result_dict["teeth_by_group"][group].append(detection)
                
                tooth_id += 1
                
            except Exception as e:
                print(f"[ERROR] Failed to classify tooth {box_idx}: {e}")
                continue

        return result_dict

    def process_image(self, image_path: str, **kwargs) -> Dict:
        """Process image from file path"""
        image = Image.open(image_path).convert("RGB")
        return self.detect_and_classify(image, **kwargs)


def format_results(pipeline_result: Dict) -> str:
    """Format pipeline results as human-readable string"""
    lines = []
    lines.append("=" * 70)
    lines.append("TOOTH SEGMENTATION AND CLASSIFICATION RESULTS")
    lines.append("=" * 70)
    
    # Radiograph validation
    is_radio = pipeline_result.get("is_radiograph")
    radio_conf = pipeline_result.get("radiograph_confidence")
    lines.append(f"\nRadiograph Validation: {'✓ YES' if is_radio else '✗ NO'} (confidence: {radio_conf:.2%})")
    
    # Detection summary
    detections = pipeline_result.get("detections", [])
    lines.append(f"\nTotal Teeth Detected: {len(detections)}")
    
    # By group
    teeth_by_group = pipeline_result.get("teeth_by_group", {})
    lines.append("\nTeeth by Group:")
    for group in [ToothGroup.INCISOR, ToothGroup.CANINE, ToothGroup.PREMOLAR, ToothGroup.MOLAR]:
        teeth = teeth_by_group.get(group, [])
        lines.append(f"  {group.value.upper():12s}: {len(teeth):2d} teeth - {', '.join([str(t.fdi_number) for t in teeth])}")
    
    # Detailed detections
    if detections:
        lines.append("\nDetailed Detections:")
        lines.append("-" * 70)
        for det in sorted(detections, key=lambda x: x.fdi_number):
            info = det.tooth_info
            lines.append(
                f"  FDI {det.fdi_number} ({info['name']:20s}) - "
                f"Group: {det.group.value:8s} - "
                f"Confidence: {det.confidence:.3f} - "
                f"BBox: ({det.bbox[0]}, {det.bbox[1]}, {det.bbox[2]}, {det.bbox[3]})"
            )
        lines.append("-" * 70)
    
    lines.append("=" * 70)
    return "\n".join(lines)


if __name__ == "__main__":
    # Example usage (requires trained models)
    print("Multiclass Tooth Detection Pipeline")
    print(f"Available FDI tooth numbers: {get_all_fdi_numbers()}")
