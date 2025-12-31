from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Optional, List
from model import classify_image, get_models, switch_model, get_active_model

app = FastAPI(title="JUNKGAME API", version="4.0 - Multi-Model")

# --- CORS Configuration ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Configuration & Constants ---
XP_PER_ITEM = 10
XP_TO_LEVEL_UP = 100

# Labels Mapping (Extended for multi-model support)
LABEL_MAP = {
    # Original 5 classes
    "glass": "유리",
    "metal": "캔",
    "plastic": "플라스틱",
    "paper": "종이",
    "waste": "일반",
    "trash": "일반",
    "cardboard": "종이",
    
    # Extended classes for 12-category model
    "battery": "배터리",
    "biological": "음식물",
    "brown-glass": "유리",
    "green-glass": "유리",
    "white-glass": "유리",
    "clothes": "의류",
    "shoes": "신발",
}

# --- Data Models ---
class CharacterStats(BaseModel):
    level: int
    xp: int
    count: int

class DetectionResult(BaseModel):
    detected_label: Optional[str]
    character_key: Optional[str]
    confidence: float
    message: str
    mask: Optional[List[List[float]]] = None
    model: Optional[str] = None
    leveled_up: bool = False
    new_stats: Optional[CharacterStats] = None

# --- Game State ---
game_state: Dict[str, CharacterStats] = {
    "일반": CharacterStats(level=1, xp=0, count=0),
    "플라스틱": CharacterStats(level=1, xp=0, count=0),
    "캔": CharacterStats(level=1, xp=0, count=0),
    "유리": CharacterStats(level=1, xp=0, count=0),
    "종이": CharacterStats(level=1, xp=0, count=0),
    "배터리": CharacterStats(level=1, xp=0, count=0),
    "음식물": CharacterStats(level=1, xp=0, count=0),
    "의류": CharacterStats(level=1, xp=0, count=0),
    "신발": CharacterStats(level=1, xp=0, count=0),
}

# --- Routes ---

@app.get("/")
def read_root():
    return {
        "status": "active", 
        "version": "4.0",
        "active_model": get_active_model(),
        "features": ["multi-model", "segmentation", "detection"]
    }

@app.get("/models")
def list_models():
    """List all available models"""
    return {
        "models": get_models(),
        "active": get_active_model()
    }

@app.post("/switch_model")
async def switch_active_model(model_key: str):
    """Switch to a different model"""
    try:
        result = switch_model(model_key)
        return {
            "success": True,
            **result,
            "message": f"Switched to {model_key}"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/stats")
def get_stats():
    return game_state

@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    """Real-time tracking endpoint with segmentation."""
    try:
        contents = await file.read()
        result = classify_image(contents)
        
        if not result:
            return {
                "detected_label": None,
                "character_key": None,
                "confidence": 0,
                "message": "인식 대기 중...",
                "mask": None,
                "model": get_active_model()
            }
        
        label = result['label'].lower()
        score = result['score']
        mask = result.get('mask', None)
        model_name = result.get('model', get_active_model())
        
        character_key = LABEL_MAP.get(label, "일반")
        
        return {
            "detected_label": label,
            "character_key": character_key,
            "confidence": score,
            "message": f"{character_key} 감지됨",
            "mask": mask,
            "model": model_name
        }
    except Exception as e:
        print(f"Error in predict: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

@app.post("/classify")
async def classify_and_collect_endpoint(file: UploadFile = File(...)):
    """Gameplay endpoint with segmentation."""
    try:
        contents = await file.read()
        result = classify_image(contents)
        
        if not result:
            return {
                "detected_label": None,
                "message": "물체를 찾을 수 없습니다.",
                "mask": None,
                "model": get_active_model()
            }
        
        label = result['label'].lower()
        score = result['score']
        mask = result.get('mask', None)
        model_name = result.get('model', get_active_model())
        character_key = LABEL_MAP.get(label, "일반")
        
        target_key = character_key
        if target_key not in game_state:
            # Auto-create new category if needed
            game_state[target_key] = CharacterStats(level=1, xp=0, count=0)
            
        stats = game_state[target_key]
        stats.count += 1
        stats.xp += XP_PER_ITEM
        
        leveled_up = False
        if stats.xp >= stats.level * XP_TO_LEVEL_UP:
            stats.level += 1
            stats.xp = 0 
            leveled_up = True
            
        return {
            "detected_label": label,
            "character_key": target_key,
            "confidence": score,
            "leveled_up": leveled_up,
            "new_stats": stats,
            "message": f"{target_key} 획득!",
            "mask": mask,
            "model": model_name
        }
    except Exception as e:
        print(f"Error in classify: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}
    
@app.post("/reset")
def reset_game():
    global game_state
    for key in game_state:
        game_state[key] = CharacterStats(level=1, xp=0, count=0)
    return {"message": "Game reset successfully"}
