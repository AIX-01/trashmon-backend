from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from model import classify_image, generate_trash_monster
import base64
from io import BytesIO

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
        
        # Map English label to Korean category
        category = LABEL_MAP.get(label, "일반쓰레기")
        
        # Get recycling guide for this category
        guide = RECYCLING_GUIDES.get(category, RECYCLING_GUIDES["일반쓰레기"])
        
        # Get box if available
        box = result.get('box')
        
        # Generate monster image using SD-Turbo model
        monster_name = f"{category} 몬스터"
        monster_image = None
        
        try:
            print(f"Generating monster image for category: {category}")
            monster_image_pil = generate_trash_monster(contents, category)
            
            if monster_image_pil:
                # Convert PIL Image to base64 data URI
                buffered = BytesIO()
                monster_image_pil.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                monster_image = f"data:image/png;base64,{img_str}"
                print("Monster image generated successfully!")
            else:
                print("Monster generation returned None, using fallback")
                monster_image = None
        except Exception as e:
            print(f"Error generating monster: {e}")
            monster_image = None
        
        # Fallback: create simple placeholder if generation failed
        if not monster_image:
            from PIL import Image, ImageDraw
            
            print("Using fallback placeholder monster image")
            img = Image.new('RGBA', (200, 200), (255, 255, 255, 0))
            draw = ImageDraw.Draw(img)
            
            category_color = guide.get('monster_color', '#78909C')
            hex_color = category_color.lstrip('#')
            rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            
            draw.ellipse([30, 30, 170, 170], fill=rgb + (255,))
            draw.ellipse([70, 70, 90, 90], fill=(0, 0, 0, 255))
            draw.ellipse([110, 70, 130, 90], fill=(0, 0, 0, 255))
            draw.arc([60, 100, 140, 140], start=0, end=180, fill=(0, 0, 0, 255), width=3)
            
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            monster_image = f"data:image/png;base64,{img_str}"

        return {
            "detected_label": label,
            "character_key": character_key,
            "confidence": score,
            "monster_name": monster_name,
            "monster_image": monster_image,
            "guide": guide,
            "box": box
        }
    except Exception as e:
        print(f"Error in predict: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


