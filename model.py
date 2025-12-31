from ultralytics import YOLO
from PIL import Image
import io
import numpy as np
import warnings
from typing import Dict, Optional, List

# Suppress warnings
warnings.filterwarnings("ignore")

# Available Models Configuration
AVAILABLE_MODELS = {
    'yolov8-segment': {
        'repo': 'turhancan97/yolov8-segment-trash-detection',
        'filename': 'yolov8m-seg.pt',
        'type': 'segmentation',
        'description': 'YOLOv8 Segmentation - 5 classes (COCO + TrashNet + TACO)',
        'classes_map': {
            0: 'Glass', 1: 'Metal', 2: 'Paper', 3: 'Plastic', 4: 'Waste'
        }
    }
}

class MultiModelTrashClassifier:
    def __init__(self, default_model='yolov8-segment'):
        self.models = {}  # Cache loaded models
        self.active_model_key = default_model
        self.active_model = None
        self.load_model(default_model)

    def get_available_models(self) -> Dict:
        """Return list of available models with metadata"""
        return {
            key: {
                'description': config['description'],
                'type': config['type'],
                'classes': config.get('classes_map', {})
            }
            for key, config in AVAILABLE_MODELS.items()
        }

    def load_model(self, model_key: str):
        """Load a specific model by key"""
        if model_key not in AVAILABLE_MODELS:
            raise ValueError(f"Model '{model_key}' not found. Available: {list(AVAILABLE_MODELS.keys())}")
        
        # Check cache
        if model_key in self.models:
            print(f"Using cached model: {model_key}")
            self.active_model = self.models[model_key]
            self.active_model_key = model_key
            return

        config = AVAILABLE_MODELS[model_key]
        
        try:
            print(f"Loading model: {model_key} ({config['description']})...")
            from huggingface_hub import hf_hub_download
            
            # Download weights
            model_path = hf_hub_download(
                repo_id=config['repo'],
                filename=config['filename']
            )
            
            print(f"Model downloaded to: {model_path}")
            
            # Load model
            model = YOLO(model_path)
            
            # Update classes map if not preset
            if not config['classes_map']:
                config['classes_map'] = model.names
            
            print(f"Model loaded successfully. Classes: {model.names}")
            
            # Cache the model
            self.models[model_key] = model
            self.active_model = model
            self.active_model_key = model_key
            
        except Exception as e:
            print(f"ERROR loading model '{model_key}': {e}")
            import traceback
            traceback.print_exc()
            raise

    def switch_model(self, model_key: str):
        """Switch to a different model"""
        print(f"Switching from '{self.active_model_key}' to '{model_key}'...")
        self.load_model(model_key)

    def predict(self, image_bytes: bytes, threshold: float = 0.5):
        if not self.active_model:
            return None

        try:
            image = Image.open(io.BytesIO(image_bytes))
            
            # Perform inference
            results = self.active_model(image, conf=threshold, verbose=False)
            
            if not results or len(results) == 0:
                return None
            
            result = results[0]
            
            # Check if there are any detections
            if result.boxes is None or len(result.boxes) == 0:
                return None
            
            # Get the best detection (highest confidence)
            confidences = result.boxes.conf.cpu().numpy()
            best_idx = np.argmax(confidences)
            
            best_conf = float(confidences[best_idx])
            best_class = int(result.boxes.cls[best_idx].cpu().numpy())
            label = self.active_model.names[best_class]
            
            # Extract mask if available (segmentation models)
            mask_polygon = None
            model_config = AVAILABLE_MODELS[self.active_model_key]
            
            if model_config['type'] == 'segmentation' and result.masks is not None and len(result.masks) > 0:
                mask = result.masks[best_idx]
                
                if hasattr(mask, 'xy') and len(mask.xy) > 0:
                    polygon = mask.xy[0]
                    mask_polygon = [[float(x), float(y)] for x, y in polygon]
            
            return {
                'label': label,
                'score': best_conf,
                'mask': mask_polygon,
                'model': self.active_model_key
            }

        except Exception as e:
            print(f"Inference Error: {e}")
            import traceback
            traceback.print_exc()
            return None

# Singleton Instance
classifier = MultiModelTrashClassifier(default_model='yolov8-segment')

def classify_image(image_bytes: bytes):
    """Wrapper function to maintain compatibility"""
    return classifier.predict(image_bytes)

def get_models():
    """Get available models"""
    return classifier.get_available_models()

def switch_model(model_key: str):
    """Switch active model"""
    classifier.switch_model(model_key)
    return {'active_model': model_key}

def get_active_model():
    """Get current active model"""
    return classifier.active_model_key
