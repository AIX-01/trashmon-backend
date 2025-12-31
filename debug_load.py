
import yolov5
import torch
import sys

print("Attempting to load with updated safe globals...")

try:
    # Try to import the class that was blocked
    # It might be under yolov5.models.yolo
    from yolov5.models.yolo import DetectionModel
    # Also might need others depending on the error
    
    torch.serialization.add_safe_globals([DetectionModel])
    print("Added DetectionModel to safe globals.")
    
    model = yolov5.load('turhancan97/yolov5-detect-trash-classification')
    print("Loaded successfully!")
    print("Classes:", model.names)
    
except ImportError:
    print("Could not import DetectionModel directly. Trying to infer location...")
    # Inspect yolov5 package structure if possible
except Exception as e:
    print(f"Error: {e}")
