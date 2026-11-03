"""스킬 스크립트 공통: 저장소 src 를 불러오고 참조 파일을 읽는다."""
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
SKILL = ROOT / "skills" / "credit-score-ops"
OPS = ROOT / "ops"
REF_DIR = OPS / "reference"
REASON_KO = {
    "delinq_now": "이번 달 연체", "delinq_max_6m": "최근 6개월 최장 연체", "delinq_months_6m": "6개월 중 연체한 달 수",
    "delinq_months_3m": "3개월 중 연체한 달 수", "idle_months_6m": "카드 미사용 달 수", "util_now": "이번 달 한도 소진율",
    "util_avg_6m": "6개월 평균 한도 소진율", "util_max_6m": "6개월 최고 한도 소진율", "bill_trend_6m": "청구액 증가 추세",
    "pay_ratio_avg_5m": "청구 대비 납부 비율(5개월 평균)", "pay_ratio_now": "직전 청구 대비 납부 비율",
    "zero_pay_months_5m": "청구가 있는데 0원 낸 달 수", "paid_to_limit_6m": "한도 대비 납부액", "limit_bal": "카드 한도",
    "education_grp": "학력 구분",
}
PROTECTED = ["sex", "age"]   # 모형에 쓰지 않고 점검에만


def thresholds() -> dict:
    return json.loads((SKILL / "references" / "thresholds.json").read_text(encoding="utf-8"))


def load_reference() -> dict:
    with open(REF_DIR / "reference.pkl", "rb") as f:
        return pickle.load(f)
