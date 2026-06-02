import asyncio
import math
import os
import uuid
from io import BytesIO
from pathlib import Path
from typing import List, Optional

import cv2
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

ROW_Y_TOLERANCE = 10  # 같은 행으로 묶는 Y축 픽셀 오차 범위


# ---------------------------------------------------------------------------
# Pydantic models — Structured Outputs + 응답 스키마
# ---------------------------------------------------------------------------

class BoundingBox(BaseModel):
    coords: List[int]  # [X1, Y1, X2, Y2]


class ItemDetail(BaseModel):
    name: str
    name_box: BoundingBox
    quantity: int
    price: int       # 단가(개당 가격)
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
# Step 1: OpenCV 전처리 — Deskew + CLAHE 대비 향상
# ---------------------------------------------------------------------------

def _deskew(img_bgr: np.ndarray) -> np.ndarray:
    """HoughLinesP로 기울기를 감지하여 이미지를 수평 보정합니다."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, math.pi / 180,
                            threshold=80, minLineLength=100, maxLineGap=10)
    if lines is None:
        return img_bgr

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 != x1:
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            if -45 < angle < 45:  # 수평에 가까운 선만 사용
                angles.append(angle)

    if not angles or abs(np.median(angles)) < 0.5:
        return img_bgr

    median_angle = float(np.median(angles))
    h, w = img_bgr.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), median_angle, 1.0)
    return cv2.warpAffine(img_bgr, M, (w, h),
                          flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def _enhance_contrast(img_bgr: np.ndarray) -> np.ndarray:
    """LAB 색공간에서 CLAHE를 적용해 흐릿한 글씨를 선명하게 합니다."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = cv2.merge([clahe.apply(l), a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def _preprocess(image: Image.Image) -> tuple[np.ndarray, Image.Image]:
    """
    PIL Image → Deskew → CLAHE →
    (OCR용 RGB ndarray, 하이라이팅용 PIL Image) 반환
    """
    img_bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    processed = _enhance_contrast(_deskew(img_bgr))
    rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
    return rgb, Image.fromarray(rgb)


# ---------------------------------------------------------------------------
# Step 2: PaddleOCR 텍스트·좌표 추출
# ---------------------------------------------------------------------------

def _run_ocr(img_rgb: np.ndarray) -> list[dict]:
    """PaddleOCR로 모든 텍스트 블록과 axis-aligned 픽셀 좌표를 추출합니다."""
    result = ocr_engine.ocr(img_rgb, cls=True)
    blocks = []
    if result and result[0]:
        for line in result[0]:
            quad, (text, _conf) = line
            xs = [p[0] for p in quad]
            ys = [p[1] for p in quad]
            blocks.append({
                "coords": [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))],
                "text": text,
            })
    return blocks


# ---------------------------------------------------------------------------
# Step 3: Y축 행(Row) 정렬 — 같은 줄 묶기 → X축 정렬
# ---------------------------------------------------------------------------

def _sort_into_rows(blocks: list[dict]) -> list[list[dict]]:
    """
    OCR 블록을 Y1 좌표 오차범위(ROW_Y_TOLERANCE) 내에서 같은 행으로 묶고,
    각 행 내부를 X1 기준 왼→오른 순서로 정렬합니다.
    """
    if not blocks:
        return []

    sorted_by_y = sorted(blocks, key=lambda b: b["coords"][1])
    rows: list[list[dict]] = []
    current_row = [sorted_by_y[0]]
    current_y = sorted_by_y[0]["coords"][1]

    for block in sorted_by_y[1:]:
        block_y = block["coords"][1]
        if abs(block_y - current_y) <= ROW_Y_TOLERANCE:
            current_row.append(block)
        else:
            rows.append(sorted(current_row, key=lambda b: b["coords"][0]))
            current_row = [block]
            current_y = block_y

    rows.append(sorted(current_row, key=lambda b: b["coords"][0]))
    return rows


def _format_for_llm(rows: list[list[dict]]) -> str:
    """정렬된 행 데이터를 LLM 프롬프트용 구조화 텍스트로 변환합니다."""
    lines = []
    for row in rows:
        row_str = " | ".join(
            f'{b["text"]} (coords:{b["coords"]})'
            for b in row
        )
        lines.append(row_str)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Step 4: LLM 호출 (1심 / 2심 공용)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "당신은 영수증 데이터 추출 전문가입니다. "
    "OCR 텍스트와 좌표를 분석해 지정된 JSON 스키마에 맞게 정확히 추출하세요. "
    "coords는 반드시 해당 텍스트 블록의 coords 값을 그대로 사용하고, "
    "price는 품목의 단가(개당 가격)입니다."
)

_USER_PROMPT_TEMPLATE = (
    "아래는 영수증 OCR 결과입니다.\n"
    "형식: 텍스트 (coords:[X1,Y1,X2,Y2]) — 같은 줄의 요소는 ' | '로 구분됩니다.\n\n"
    "{ocr_data}\n\n"
    "상호명, 결제일시(YYYY-MM-DD HH:MM:SS), "
    "세부 결제 내역(상품명·수량·단가), 총 결제금액을 추출하세요."
)


async def _call_llm(ocr_data: str, model: str) -> ReceiptData:
    response = await client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _USER_PROMPT_TEMPLATE.format(ocr_data=ocr_data)},
        ],
        response_format=ReceiptData,
        max_tokens=2048,
    )
    parsed = response.choices[0].message.parsed
    if parsed is None:
        raise ValueError(f"{model}이 구조화된 데이터를 반환하지 않았습니다.")
    return parsed


# ---------------------------------------------------------------------------
# Step 5: 무결성 검증 + 2심 재시도
# ---------------------------------------------------------------------------

def _verify_total(data: ReceiptData) -> bool:
    """sum(수량 × 단가) == total_amount 일치 여부를 검증합니다."""
    return sum(item.quantity * item.price for item in data.items) == data.total_amount


async def _extract_verified(ocr_data: str) -> ReceiptData:
    """
    1심: gpt-4o-mini → 검증 통과 시 반환.
    검증 실패 시 2심: gpt-4o 재시도 후 최선 결과 반환.
    """
    data = await _call_llm(ocr_data, model="gpt-4o-mini")
    if _verify_total(data):
        return data

    # 2심 — 상위 모델로 재추출
    data = await _call_llm(ocr_data, model="gpt-4o")
    return data


# ---------------------------------------------------------------------------
# Step 6: Pillow 하이라이팅 & 저장
# ---------------------------------------------------------------------------

def _draw_highlights(image: Image.Image, data: ReceiptData) -> Image.Image:
    """최종 확정 좌표 위에 반투명 하이라이트 박스를 합성합니다."""
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


# ---------------------------------------------------------------------------
# API 엔드포인트
# ---------------------------------------------------------------------------

@app.post("/api/analyze-receipt", response_model=AnalyzeResponse)
async def analyze_receipt(file: UploadFile = File(...)):
    """
    OpenCV → PaddleOCR → Y축 정렬 → gpt-4o-mini(1심) → 검증 → gpt-4o(2심)
    초고정밀 하이브리드 파이프라인. 분석 불가 시 success: false 반환.
    """
    try:
        raw_image = Image.open(BytesIO(await file.read()))

        # 1. OpenCV 전처리 (Deskew + CLAHE) — 블로킹이므로 스레드풀 실행
        img_array, processed_image = await asyncio.to_thread(_preprocess, raw_image)

        # 2. PaddleOCR 추출 — 블로킹이므로 스레드풀 실행
        ocr_blocks = await asyncio.to_thread(_run_ocr, img_array)

        # 3. Y축 행 정렬 → LLM 입력 텍스트 생성
        rows = _sort_into_rows(ocr_blocks)
        ocr_data = _format_for_llm(rows)

        # 4. 1심(gpt-4o-mini) + 무결성 검증 + 필요 시 2심(gpt-4o)
        data = await _extract_verified(ocr_data)

        # 5. 하이라이팅 & 저장
        highlighted = _draw_highlights(processed_image, data)
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
