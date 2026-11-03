"""신청자 한 명의 점수 설명: 피처별 점수 · 받을 수 있는 최고 점수와의 차이 · 사유 코드. 결정은 내리지 않는다.
uv run python skills/credit-score-ops/scripts/explain.py --scores ops/runs/2026-01/scores.csv --id 123
"""
import argparse
import json

import pandas as pd

from _common import REASON_KO, load_reference


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--scores", required=True); ap.add_argument("--id", type=int, required=True)
    a = ap.parse_args(); ref = load_reference(); card = ref["card"]
    sc = pd.read_csv(a.scores); row = sc[sc["id"] == a.id]
    if row.empty:
        raise SystemExit(f"id {a.id} 가 점수 파일에 없음")
    row = row.iloc[0]; best = card.table.groupby("feature")["points"].max()
    parts = []
    for f in card.features:
        got = int(row[f"pts_{f}"]); parts.append({"feature": REASON_KO.get(f, f), "points": got, "max_points": int(best[f]), "gap": int(best[f]) - got})
    parts.sort(key=lambda x: -x["gap"])
    out = {"id": a.id, "score": float(row["score"]), "cut_score": ref["cut_score"], "vs_cut": row["vs_cut"], "band": int(row["band"]),
           "prob_default": float(row["prob_default"]), "reasons": [row["reason_1"], row["reason_2"], row["reason_3"]], "points": parts,
           "note": "사유 코드 = 그 피처에서 받을 수 있는 최고 점수와의 차이가 큰 순서. 승인 · 거절은 심사 담당자가 정한다"}
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
