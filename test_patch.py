
import torch
import yolov5

# Monkey patch torch.load to always use weights_only=False
_original_load = torch.load

def unsafe_load(*args, **kwargs):
    # Force weights_only=False
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)

torch.load = unsafe_load
print("Monkey patched torch.load to be unsafe (weights_only=False).")

try:
    model = yolov5.load('turhancan97/yolov5-detect-trash-classification')
    print("Model loaded successfully!")
    print("Classes:", model.names)
except Exception as e:
    print(f"Error: {e}")
