# 🧾 Vision LLM 기반 영수증 분석 및 하이라이트 API 서버

기존 Donut 모델(`naver-clova-ix/donut-base-finetuned-cord-v2`)의 정형화되지 않은 영수증(Wild Receipt)에 대한 인식률 저하 문제를 해결하기 위해, **Multi-modal Vision LLM(OpenAI gpt-4o-mini)**을 도입한 FastAPI 기반의 영수증 데이터 추출 및 시각화 AI 서버입니다.

---

## 🛠️ 기술 스택 (Tech Stack)

- **Language:** Python 3.11.9 (amd64)
- **Framework:** FastAPI (Uvicorn)
- **AI/ML:** PyTorch, Transformers (Hugging Face) ※ 기존 인프라 유지용
- **Vision Engine:** OpenAI API (`gpt-4o-mini`), Pillow (PIL)

---

## 📌 주요 변경 사항 (Migration From Donut)

1. **엔진 스위칭:** 무겁고 한국어 학습 데이터가 부족했던 Donut 모델 대신, 대규모 멀티모달 모델인 `gpt-4o-mini`를 사용하여 제각각인 영수증 레이아웃에 유연하게 대응합니다.
2. **공간 정보(Coordinates) 활용:** 텍스트 추출과 동시에 각 항목의 픽셀 좌표`[X1, Y1, X2, Y2]`를 함께 확보합니다.
3. **바탕화면 저장 (Highlighting):** 추출된 좌표를 기반으로 원본 이미지 위에 반투명 박스를 그린 뒤, **서버를 실행한 사용자의 PC 바탕화면(Desktop)에 UUID를 파일명으로 하여 실물 이미지 파일로 저장**합니다.

---

## 📋 데이터 추출 및 제약 조건

OpenAI의 **Structured Outputs** 기능을 강제하여 아래 구조의 데이터 타입을 완벽하게 보장합니다.

- **상호명** (가게 이름) + 좌표
- **결제일시** (날짜 및 시간, `YYYY-MM-DD HH:MM:SS` 포맷 선호) + 좌표
- **세부 결제 내역 리스트** (상품명, 수량, 가격) + 각 항목별 좌표
- **총 결제금액** + 좌표

---

## 🚀 API 명세 (API Specification)

### 영수증 분석 및 하이라이트 생성

인증 처리는 전면의 Spring 웹 서버(세션 방식)에서 수행하므로 본 API는 순수 분석 기능만 제공합니다. 영수증 훼손 등으로 분석 실패 시 에러를 던지지 않고 `success: false` 상태값을 반환합니다.

- **URL:** `/analyze-receipt`
- **Method:** `POST`
- **Content-Type:** `multipart/form-data`

#### Request Parameters

| 파라미터명 | 타입         | 필수 여부 | 설명                                          |
| :--------- | :----------- | :-------- | :-------------------------------------------- |
| `file`     | File (Image) | 필수      | 분석할 영수증 이미지 파일 (png, jpg, jpeg 등) |

#### Response Example (성공 시 - `200 OK`)

_성공 시 사용자의 PC 바탕화면에 하이라이트 이미지가 실제로 생성되며, 응답 결과로 해당 절대 경로를 반환합니다._

```json
{
  "success": true,
  "file_path": "C:\\Users\\Lee\\Desktop\\upload\\receipt_b1a2c3d4-e5f6-7a8b-9c0d-1e2f3a4b5c6d.jpg",
  "data": {
    "store_name": "스타벅스 강남점",
    "store_name_box": { "coords": [120, 45, 340, 85] },
    "payment_date": "2026-06-02 14:20:15",
    "payment_date_box": { "coords": [50, 110, 280, 135] },
    "items": [
      {
        "name": "아메리카노",
        "name_box": { "coords": [50, 200, 180, 225] },
        "quantity": 1,
        "price": 4500,
        "price_box": { "coords": [350, 200, 410, 225] }
      },
      {
        "name": "아이스라떼",
        "name_box": { "coords": [50, 230, 180, 255] },
        "quantity": 2,
        "price": 10000,
        "price_box": { "coords": [350, 230, 410, 255] }
      }
    ],
    "total_amount": 14500,
    "total_amount_box": { "coords": [330, 450, 420, 480] }
  }
}
```

위의 스펙을 충족하는 완성도 높은 `main.py` 전체 코드를 작성해 주세요. (.env 파일에서 OpenAI 키를 로드하는 코드도 포함해 주세요.)
