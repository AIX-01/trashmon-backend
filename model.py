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
                
                if hasattr(mask, 'xyn') and len(mask.xyn) > 0:
                    polygon = mask.xyn[0]
                    mask_polygon = [[float(x), float(y)] for x, y in polygon]
            
            # Extract bounding box (normalized coordinates)
            box = result.boxes.xyxyn[best_idx].cpu().numpy().tolist()
            
            return {
                'label': label,
                'score': best_conf,
                'mask': mask_polygon,
                'box': box,  # [x1, y1, x2, y2] normalized 0~1
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

def classify_image_multiframe(images_bytes: List[bytes], threshold: float = 0.5):
    """
    Classify multiple frames and return the most confident and frequently detected object.
    
    Args:
        images_bytes: List of image bytes (typically 6 frames captured over 0.5 seconds)
        threshold: Confidence threshold for detections
    
    Returns:
        Best classification result with highest frequency and confidence
    """
    if not images_bytes or len(images_bytes) == 0:
        return None
    
    # Collect all predictions
    predictions = []
    for img_bytes in images_bytes:
        result = classifier.predict(img_bytes, threshold=threshold)
        if result:
            predictions.append(result)
    
    if not predictions:
        return None
    
    # Count frequency of each label
    from collections import Counter
    label_counts = Counter([pred['label'] for pred in predictions])
    
    # Find the most frequent label(s)
    max_count = max(label_counts.values())
    most_frequent_labels = [label for label, count in label_counts.items() if count == max_count]
    
    # Among the most frequent labels, find the one with highest confidence
    best_prediction = None
    best_score = 0.0
    
    for pred in predictions:
        if pred['label'] in most_frequent_labels and pred['score'] > best_score:
            best_score = pred['score']
            best_prediction = pred
    
    return best_prediction

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

# ==================== Monster Generation ====================

# 공통 스타일 프롬프트 (단순하고 귀여운 스티커 스타일, 흰색 배경)
COMMON_STYLE = """cute 2D cartoon cat character illustration,
flat vector style, pastel colors,
full body view of a single anthropomorphic cat character,
round body with big head and small body,
short arms and legs,
simple facial expression with dot eyes and small mouth, cheerful and friendly mood, character fully visible and centered,
various cute poses, solid color background, minimal details, no shading, no close-up, no cropped body, no zoom"""

# 한글 프롬프트 템플릿 (분리수거 카테고리별)
MONSTER_PROMPTS = {
    "플라스틱": f"{COMMON_STYLE}",
    "종이": f"{COMMON_STYLE}",
    "유리": f"{COMMON_STYLE}",
    "캔": f"{COMMON_STYLE}",
    "일반쓰레기": f"{COMMON_STYLE}"
}

class MonsterGenerator:
    def __init__(self):
        self.pipe = None
        self.device = None
        self._load_model()
    
    def _load_model(self):
        """SD-Turbo 모델 로드"""
        try:
            print("Loading SD-Turbo model for monster generation...")
            from diffusers import AutoPipelineForImage2Image
            import torch
            
            # GPU 사용 가능 여부 확인
            if torch.cuda.is_available():
                self.device = "cuda"
                print(f"GPU detected: {torch.cuda.get_device_name(0)}")
            else:
                self.device = "cpu"
                print("GPU not available, using CPU (will be slower)")
            
            # SD-Turbo 모델 로드 (경량화 버전)
            self.pipe = AutoPipelineForImage2Image.from_pretrained(
                "stabilityai/sd-turbo",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                variant="fp16" if self.device == "cuda" else None
            )
            
            self.pipe.to(self.device)
            
            # 메모리 최적화
            if self.device == "cuda":
                self.pipe.enable_attention_slicing()
            
            print("Monster generator ready!")
            
        except Exception as e:
            print(f"Error loading monster generator: {e}")
            import traceback
            traceback.print_exc()
            self.pipe = None
    
    def generate_monster(self, image_bytes: bytes, category: str = "일반쓰레기", strength: float = 0.8):
        """
        쓰레기 이미지를 귀여운 괴물로 변환
        1. YOLO Segmentation을 사용하여 입력 이미지에서 배경 제거
        2. 흰색 배경에 합성하여 SD-Turbo에 입력
        3. 생성된 이미지에서 흰색 배경 제거 (투명 괴물)
        """
        if not self.pipe:
            print("Monster generator not available!")
            return None
        
        try:
            # 1. 이미지 로드
            input_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            
            # 2. 배경 제거 (Input) - YOLO Segmentation 사용
            processed_input = input_image
            try:
                print("Isolating object using YOLO Segmentation...")
                # 전역 classifier 인스턴스 사용
                from model import classifier
                
                # YOLO 추론
                result = classifier.predict(image_bytes, threshold=0.3)
                
                if result and result.get('mask'):
                    import cv2
                    import numpy as np
                    
                    # 마스크 폴리곤 가져오기 (정규화된 좌표)
                    polygon = result['mask'] # [[x,y], [x,y], ...]
                    
                    if polygon:
                        w, h = input_image.size
                        
                        # 폴리곤을 픽셀 좌표로 변환
                        pixel_polygon = np.array([[int(p[0] * w), int(p[1] * h)] for p in polygon], dtype=np.int32)
                        
                        # 마스크 이미지 생성 (검은 배경)
                        mask_img = np.zeros((h, w), dtype=np.uint8)
                        # 폴리곤 영역을 흰색으로 채움
                        cv2.fillPoly(mask_img, [pixel_polygon], 255)
                        
                        # PIL 이미지로 변환
                        mask_pil = Image.fromarray(mask_img)
                        
                        # 흰색 배경 이미지 생성
                        white_bg = Image.new("RGB", (w, h), (255, 255, 255))
                        
                        # 마스크를 사용하여 원본 이미지와 흰색 배경 합성
                        # 마스크가 흰색인 부분(객체)은 원본, 검은색(배경)은 흰색 배경
                        processed_input = Image.composite(input_image, white_bg, mask_pil)
                        print("Object isolated successfully using YOLO mask.")
                    else:
                        print("No mask polygon found.")
                else:
                    print("No object detected or no mask available.")
                    
            except Exception as e:
                print(f"Input segmentation failed: {e}, using original")
                processed_input = input_image
            
            # 3. 리사이즈 (SD-Turbo 최적화)
            processed_input = processed_input.resize((512, 512))
            
            # 4. 프롬프트 설정 (흰색 배경 강조)
            base_prompt = MONSTER_PROMPTS.get(category, MONSTER_PROMPTS["일반쓰레기"])
            prompt = f"{base_prompt}, white background"
            
            print(f"Generating {category} monster with prompt: {prompt[:50]}...")
            
            # 5. 이미지 생성
            generated_image = self.pipe(
                prompt=prompt,
                image=processed_input,
                num_inference_steps=2,
                strength=strength,
                guidance_scale=0.0
            ).images[0]
            
            print("Monster generated successfully!")
            
            # 6. 배경 투명화 (Output) - Flood Fill 사용
            final_image = generated_image
            try:
                import cv2
                import numpy as np
                
                # PIL -> OpenCV (RGB -> BGR)
                img_np = np.array(generated_image)
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                
                h, w = img_bgr.shape[:2]
                
                # 마스크 생성 (h+2, w+2) - floodFill 요구사항
                mask = np.zeros((h+2, w+2), np.uint8)
                
                # 홍수 채우기 (Flood Fill) - (0,0) 좌표에서 시작하여 흰색 배경을 투명하게 만듦
                # loDiff, upDiff: 색상 허용 오차 (흰색 배경이 완벽한 255가 아닐 수 있으므로 여유를 둠)
                diff = (10, 10, 10)
                
                # 네 귀퉁이에서 모두 시도 (배경이 끊겨 있을 수 있음)
                corners = [(0, 0), (w-1, 0), (0, h-1), (w-1, h-1)]
                
                # 투명 처리를 위해 Alpha 채널 생성
                b, g, r = cv2.split(img_bgr)
                alpha = np.full((h, w), 255, dtype=np.uint8)
                
                # 흰색 탐지 기준 (밝은 색)
                # 시작점이 충분히 밝은 경우에만 배경 제거 시도
                for x, y in corners:
                    if np.all(img_bgr[y, x] > 200): # 밝은 픽셀인 경우에만
                        # 임시 마스크 복사
                        temp_mask = mask.copy()
                        cv2.floodFill(img_bgr, temp_mask, (x, y), (0, 0, 0), diff, diff, flags=cv2.FLOODFILL_FIXED_RANGE | (255 << 8) | cv2.FLOODFILL_MASK_ONLY)
                        
                        # temp_mask에서 채워진 영역(255)을 mask에 합침
                        mask = cv2.bitwise_or(mask, temp_mask)

                # 마스크 크기 복원 (가장자리 제거)
                mask = mask[1:-1, 1:-1]
                
                # 안전 장치: 너무 많은 영역이 지워졌는지 확인 (95% 이상이면 경고)
                total_pixels = w * h
                removed_pixels = np.count_nonzero(mask)
                removal_ratio = removed_pixels / total_pixels
                
                print(f"Background removal ratio: {removal_ratio:.2f}")
                
                if removal_ratio > 0.95:
                    print("First pass removed too much (>95%). Trying stricter tolerance...")
                    
                    # 2차 시도: 더 엄격한 기준 (tolerance 2)
                    mask = np.zeros((h+2, w+2), np.uint8)
                    strict_diff = (2, 2, 2)
                    
                    for x, y in corners:
                        if np.all(img_bgr[y, x] > 240): # 아주 밝은 픽셀만 시작점으로
                            temp_mask = mask.copy()
                            cv2.floodFill(img_bgr, temp_mask, (x, y), (0, 0, 0), strict_diff, strict_diff, flags=cv2.FLOODFILL_FIXED_RANGE | (255 << 8) | cv2.FLOODFILL_MASK_ONLY)
                            mask = cv2.bitwise_or(mask, temp_mask)
                            
                    mask = mask[1:-1, 1:-1]
                    removed_pixels = np.count_nonzero(mask)
                    removal_ratio = removed_pixels / total_pixels
                    print(f"Second pass removal ratio: {removal_ratio:.2f}")
                
                if removal_ratio > 0.98: # 여전히 거의 다 지워졌다면
                     print("Too much background removed! Reverting to original image (safety fallback).")
                     final_image = generated_image # 원본 사용 (흰색 배경)
                else:
                    # 마스크가 있는 부분(배경)을 투명하게 처리
                    alpha[mask > 0] = 0
                    
                    # 합치기
                    final_bgra = cv2.merge([r, g, b, alpha])
                    final_image = Image.fromarray(final_bgra)
                    print("Background removed successfully (Flood Fill)")
                
            except Exception as e:
                print(f"Output background removal failed: {e}")
                # 실패 시 원본 반환 (투명화 안됨)
            
            return final_image
            
        except Exception as e:
            print(f"Error generating monster: {e}")
            import traceback
            traceback.print_exc()
            return None

# 싱글톤 인스턴스
try:
    monster_generator = MonsterGenerator()
except Exception as e:
    print(f"Failed to initialize monster generator: {e}")
    monster_generator = None

def generate_trash_monster(image_bytes: bytes, category: str = "일반쓰레기"):
    """괴물 생성 래퍼 함수"""
    if not monster_generator:
        return None
    return monster_generator.generate_monster(image_bytes, category)

