from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Optional, List
from model import classify_image

app = FastAPI(title="Trashmon Classifier API", version="1.0")

# --- CORS Configuration ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Recycling Guide Data ---
RECYCLING_GUIDES = {
    "종이": {
        "bin_color": "파란색",
        "message": "종이는 파란색 통에 쏙!",
        "tips": ["물에 젖지 않게, 테이프는 떼고 버려요."],
        "monster_color": "#4A90D9"
    },
    "유리": {
        "bin_color": "초록색",
        "message": "유리병은 초록색 통에 쏙!",
        "tips": ["뚜껑을 떼고, 안을 한번 헹궈서 버려요."],
        "monster_color": "#7CB342"
    },
    "플라스틱": {
        "bin_color": "노란색",
        "message": "플라스틱은 노란색 통에 쏙!",
        "tips": ["라벨을 떼고, 깨끗이 씻어서 버려요."],
        "monster_color": "#FFD54F"
    },
    "캔": {
        "bin_color": "빨간색",
        "message": "캔은 빨간색 통에 쏙!",
        "tips": ["납작하게 밟아서, 조심해서 버려요."],
        "monster_color": "#EF5350"
    },
    "일반쓰레기": {
        "bin_color": "검은색",
        "message": "일반쓰레기는 아무 통에나!",
        "tips": ["재활용이 어려운 친구들이에요."],
        "monster_color": "#78909C"
    }
}

# Labels Mapping (English to Korean)
LABEL_MAP = {
    "glass": "유리",
    "metal": "캔",
    "plastic": "플라스틱",
    "paper": "종이",
    "waste": "일반쓰레기",
    "trash": "일반쓰레기",
    "cardboard": "종이",
}

# --- Data Models ---
class RecyclingGuide(BaseModel):
    bin_color: str
    message: str
    tips: List[str]
    monster_color: str

class ClassificationResponse(BaseModel):
    success: bool
    category: str
    confidence: float
    guide: RecyclingGuide

# --- Routes ---

@app.get("/")
def read_root():
    return {
        "status": "active",
        "version": "1.0",
        "service": "Trashmon Classifier API"
    }

@app.post("/classify")
async def classify_endpoint(file: UploadFile = File(...)):
    """
    쓰레기 분류 엔드포인트
    이미지를 업로드하면 AI가 분류하고 분리수거 가이드를 제공합니다.
    """
    try:
        contents = await file.read()
        result = classify_image(contents)
        
        # No detection
        if not result:
            return {
                "success": False,
                "category": "",
                "confidence": 0.0,
                "guide": {
                    "bin_color": "",
                    "message": "물체를 찾을 수 없습니다.",
                    "tips": [],
                    "monster_color": "#CCCCCC"
                }
            }
        
        # Get detected label and map to Korean category
        label = result['label'].lower()
        score = result['score']
        
        # Map English label to Korean category
        category = LABEL_MAP.get(label, "일반쓰레기")
        
        # Get recycling guide for this category
        guide = RECYCLING_GUIDES.get(category, RECYCLING_GUIDES["일반쓰레기"])
        
        return {
            "success": True,
            "category": category,
            "confidence": score,
            "guide": guide
        }
        
    except Exception as e:
        print(f"Error in classify: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            "success": False,
            "category": "",
            "confidence": 0.0,
            "guide": {
                "bin_color": "",
                "message": f"오류가 발생했습니다: {str(e)}",
                "tips": [],
                "monster_color": "#CCCCCC"
            }
        }
