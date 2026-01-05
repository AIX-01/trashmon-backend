from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
from model import classify_image, generate_trash_monster
import base64
from io import BytesIO

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
            "success": True,
            "category": category,
            "confidence": score,
            "monster_name": monster_name,
            "monster_image": monster_image,
            "guide": guide,
            "box": box
        }
        
    except Exception as e:
        print(f"Error in generate-monster: {e}")
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


