import asyncio
import json
import os
import uuid
from io import BytesIO
from pathlib import Path
from typing import List, Optional

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from openai import AsyncOpenAI
from paddleocr import PaddleOCR
from PIL import Image, ImageDraw
from pydantic import BaseModel

load_dotenv()

app = FastAPI(title="영수증 분석 API")
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
ocr_engine = PaddleOCR(use_angle_cls=True, lang="korean", show_log=False, use_gpu=False)

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

def _run_ocr(image: Image.Image) -> list[dict]:
    """PaddleOCR로 이미지 내 모든 텍스트 블록과 픽셀 좌표를 추출합니다."""
    img_array = np.array(image.convert("RGB"))
    result = ocr_engine.ocr(img_array, cls=True)

    blocks = []
    if result and result[0]:
        for line in result[0]:
            quad, (text, _conf) = line
            # quad: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] → axis-aligned bbox
            xs = [p[0] for p in quad]
            ys = [p[1] for p in quad]
            blocks.append({
                "coords": [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))],
                "text": text,
            })
    return blocks


def _draw_highlights(image: Image.Image, data: ReceiptData) -> Image.Image:
    """매칭된 좌표 위에 반투명 하이라이트 박스를 합성합니다."""
    rgba = image.convert("RGBA")
    overlay = Image.new("RGBA", rgba.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    fill = (255, 220, 0, 90)
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


def _save_highlighted(image: Image.Image) -> str:
    """하이라이트 이미지를 업로드 폴더에 UUID 파일명으로 저장하고 파일명을 반환합니다."""
    file_name = f"receipt_{uuid.uuid4()}.jpg"
    image.save(str(UPLOAD_DIR / file_name), format="JPEG", quality=95)
    return file_name


async def _call_openai(ocr_blocks: list[dict]) -> ReceiptData:
    """PaddleOCR 결과(텍스트+좌표)만 LLM에 전달하여 구조화된 데이터를 추출합니다 (이미지 미전송)."""
    ocr_text = json.dumps(ocr_blocks, ensure_ascii=False)

    response = await client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": (
                    "다음은 영수증 이미지에서 OCR로 추출한 텍스트 블록 목록입니다.\n"
                    "형식: {\"coords\": [X1, Y1, X2, Y2], \"text\": \"텍스트\"}\n\n"
                    f"{ocr_text}\n\n"
                    "위 데이터를 분석하여 상호명, 결제일시(YYYY-MM-DD HH:MM:SS 형식), "
                    "세부 결제 내역(상품명·수량·가격), 총 결제금액을 추출하세요. "
                    "각 항목의 coords는 해당 OCR 블록의 coords 값을 그대로 사용하세요."
                ),
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
    PaddleOCR → gpt-4o-mini 하이브리드 파이프라인으로 영수증을 분석합니다.
    분석 실패 시 예외를 던지지 않고 success: false 를 반환합니다.
    """
    try:
        image = Image.open(BytesIO(await file.read()))
        # PaddleOCR은 동기 블로킹 → 스레드풀에서 실행
        ocr_blocks = await asyncio.to_thread(_run_ocr, image)
        data = await _call_openai(ocr_blocks)
        highlighted = _draw_highlights(image, data)
        file_name = _save_highlighted(highlighted)
        return AnalyzeResponse(success=True, file_name=file_name, data=data)
    except Exception:
        return AnalyzeResponse(success=False)


# ---------------------------------------------------------------------------
# 로컬 실행 진입점
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=7777, reload=True)
