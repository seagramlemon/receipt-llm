import os
import uuid
import base64
from io import BytesIO
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from openai import AsyncOpenAI
from PIL import Image, ImageDraw
from pydantic import BaseModel

load_dotenv()

app = FastAPI(title="영수증 분석 API")
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

UPLOAD_DIR = Path(r"C:\Users\Lee\Desktop\upload")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Pydantic models — OpenAI Structured Outputs + 응답 스키마
# ---------------------------------------------------------------------------

class BoundingBox(BaseModel):
    coords: List[int]  # [X1, Y1, X2, Y2]


class ItemDetail(BaseModel):
    name: str
    name_box: BoundingBox
    quantity: int
    price: int
    price_box: BoundingBox


class ReceiptData(BaseModel):
    store_name: str
    store_name_box: BoundingBox
    payment_date: str          # YYYY-MM-DD HH:MM:SS
    payment_date_box: BoundingBox
    items: List[ItemDetail]
    total_amount: int
    total_amount_box: BoundingBox


class AnalyzeResponse(BaseModel):
    success: bool
    file_name: Optional[str] = None
    data: Optional[ReceiptData] = None


# ---------------------------------------------------------------------------
# 내부 헬퍼 함수
# ---------------------------------------------------------------------------

def _encode_image(image: Image.Image) -> tuple[str, str]:
    """PIL Image → base64 문자열 + MIME 타입 반환."""
    fmt = image.format if image.format in ("JPEG", "PNG", "WEBP") else "JPEG"
    mime = f"image/{fmt.lower()}"
    # JPEG는 알파 채널 미지원
    src = image.convert("RGB") if fmt == "JPEG" and image.mode == "RGBA" else image
    buf = BytesIO()
    src.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode(), mime


def _draw_highlights(image: Image.Image, data: ReceiptData) -> Image.Image:
    """추출된 좌표 위에 반투명 하이라이트 박스를 합성합니다."""
    rgba = image.convert("RGBA")
    overlay = Image.new("RGBA", rgba.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    fill = (255, 220, 0, 90)      # 반투명 노란색
    outline = (255, 160, 0, 230)

    def _box(coords: List[int]) -> None:
        if coords and len(coords) == 4:
            draw.rectangle(coords, fill=fill, outline=outline, width=2)

    _box(data.store_name_box.coords)
    _box(data.payment_date_box.coords)
    _box(data.total_amount_box.coords)
    for item in data.items:
        _box(item.name_box.coords)
        _box(item.price_box.coords)

    return Image.alpha_composite(rgba, overlay).convert("RGB")


def _save_to_desktop(image: Image.Image) -> str:
    """하이라이트 이미지를 업로드 폴더에 UUID 파일명으로 저장하고 파일명을 반환합니다."""
    file_name = f"receipt_{uuid.uuid4()}.jpg"
    image.save(str(UPLOAD_DIR / file_name), format="JPEG", quality=95)
    return file_name


async def _call_openai(image: Image.Image) -> ReceiptData:
    """gpt-4o-mini Structured Outputs으로 영수증 데이터를 추출합니다."""
    b64, mime = _encode_image(image)

    response = await client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "영수증 이미지를 분석하여 다음 항목을 JSON으로 추출하세요:\n"
                            "- 상호명(store_name)과 해당 픽셀 좌표 [X1,Y1,X2,Y2]\n"
                            "- 결제일시(payment_date, 'YYYY-MM-DD HH:MM:SS' 형식 선호)와 좌표\n"
                            "- 세부 결제 내역 리스트(items): 각 항목의 상품명·수량·가격과 좌표\n"
                            "- 총 결제금액(total_amount)과 좌표\n"
                            "좌표는 이미지 내 실제 픽셀 위치로 최대한 정확하게 지정해 주세요."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime};base64,{b64}",
                            "detail": "high",
                        },
                    },
                ],
            }
        ],
        response_format=ReceiptData,
        max_tokens=2048,
    )

    parsed = response.choices[0].message.parsed
    if parsed is None:
        raise ValueError("OpenAI가 구조화된 데이터를 반환하지 않았습니다.")
    return parsed


# ---------------------------------------------------------------------------
# API 엔드포인트
# ---------------------------------------------------------------------------

@app.post("/api/analyze-receipt", response_model=AnalyzeResponse)
async def analyze_receipt(file: UploadFile = File(...)):
    """
    영수증 이미지를 분석하여 구조화된 데이터를 추출하고,
    하이라이트 이미지를 바탕화면에 저장합니다.
    분석 실패 시 예외를 던지지 않고 success: false 를 반환합니다.
    """
    try:
        image = Image.open(BytesIO(await file.read()))
        data = await _call_openai(image)
        highlighted = _draw_highlights(image, data)
        file_path = _save_to_desktop(highlighted)
        return AnalyzeResponse(success=True, file_name=file_path, data=data)
    except Exception:
        return AnalyzeResponse(success=False)


# ---------------------------------------------------------------------------
# 로컬 실행 진입점
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=7777, reload=True)
