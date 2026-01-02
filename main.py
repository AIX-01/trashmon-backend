from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Dict, Optional, List
import io
from model import classify_image, classify_image_multiframe, generate_trash_monster

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
    box: Optional[List[float]] = None  # [x1, y1, x2, y2] normalized


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
        
        # Get box if available
        box = result.get('box')

        return {
            "success": True,
            "category": category,
            "confidence": score,
            "guide": guide,
            "box": box
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
            },
            "box": None
        }

@app.post("/classify-multi")
async def classify_multiframe_endpoint(files: List[UploadFile] = File(...)):
    """
    다중 프레임 쓰레기 분류 엔드포인트
    여러 이미지를 업로드하면 AI가 가장 신뢰도 높고 많이 식별된 물체를 선택합니다.
    """
    try:
        # Read all images
        images_bytes = []
        for file in files:
            contents = await file.read()
            images_bytes.append(contents)
        
        # Perform multi-frame classification
        result = classify_image_multiframe(images_bytes)
        
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
        
        # Get box if available
        box = result.get('box')

        return {
            "success": True,
            "category": category,
            "confidence": score,
            "guide": guide,
            "frames_analyzed": len(images_bytes),
            "box": box
        }
        
    except Exception as e:
        print(f"Error in classify-multi: {e}")
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

@app.post("/generate-monster")
async def generate_monster_endpoint(
    file: UploadFile = File(...),
    category: str = Form("일반쓰레기")
):
    """
    쓰레기 이미지를 귀여운 괴물로 변환하는 엔드포인트
    
    Args:
        file: 원본 쓰레기 이미지
        category: 분리수거 카테고리 (플라스틱, 종이, 유리, 캔, 일반쓰레기)
    
    Returns:
        변환된 괴물 이미지 (PNG)
    """
    try:
        contents = await file.read()
        
        # 괴물 생성
        monster_image = generate_trash_monster(contents, category)
        
        if not monster_image:
            return {
                "success": False,
                "message": "괴물 생성에 실패했습니다. 모델이 아직 로드되지 않았을 수 있습니다."
            }
        
        # 이미지를 bytes로 변환
        img_bytes = io.BytesIO()
        monster_image.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        return StreamingResponse(img_bytes, media_type="image/png")
        
    except Exception as e:
        print(f"Error in generate-monster: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            "success": False,
            "message": f"오류가 발생했습니다: {str(e)}"
        }
