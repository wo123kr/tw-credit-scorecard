"""배치 점수 매기기: 점수 · 10구간 · 기준선 위/아래 · 거절 사유 코드 3개. 결정은 내리지 않는다(기준선 위/아래만 표시).
uv run python skills/credit-score-ops/scripts/score_batch.py --batch ops/batches/applicants_2026-01.csv --out ops/runs/2026-01
"""
import argparse
import json

import numpy as np
import pandas as pd

from _common import REASON_KO, load_reference
from card import reason_codes


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--batch", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ref = load_reference(); card = ref["card"]
    df = pd.read_csv(a.batch)
    missing = [f for f in card.features if f not in df.columns]
    if missing:
        raise SystemExit(f"피처 열이 없음: {missing}")
    # 학습 때 결측이 없던 피처에 결측이 오면 점수표에 그 칸이 없다 → 짐작으로 채우지 않고 「점수 불가」로 둔다
    no_bin = [f for f in card.features if card.binnings[f].missing_woe is None]
    bad = df[no_bin].isna().any(axis=1).to_numpy() if no_bin else np.zeros(len(df), bool)
    ok = ~bad
    score = np.full(len(df), np.nan); prob = np.full(len(df), np.nan)
    edges = np.array(ref["train_bands"]["score_min"].to_numpy(), dtype=float)
    band = np.zeros(len(df), int); pts = pd.DataFrame(index=df.index, columns=[f"pts_{f}" for f in card.features], dtype=float)
    codes = [[] for _ in range(len(df))]
    if ok.any():
        d_ok = df[ok]
        score[ok] = card.score(d_ok); prob[ok] = card.predict_bad(d_ok)
        band[ok] = np.clip(np.searchsorted(edges, score[ok], side="right"), 1, len(edges))
        p_ok = card.points(d_ok); pts.loc[ok, :] = p_ok.to_numpy()
        for i, c in zip(np.flatnonzero(ok), reason_codes(card, d_ok)):
            codes[i] = c
    vs = np.where(bad, "점수 불가(결측)", np.where(score >= ref["cut_score"], "기준선 이상", "기준선 미만"))
    out = pd.DataFrame({"id": df["id"], "score": score, "prob_default": np.round(prob, 4), "band": band, "vs_cut": vs,
                        "reason_1": [REASON_KO.get(c[0], c[0]) if len(c) > 0 else "" for c in codes],
                        "reason_2": [REASON_KO.get(c[1], c[1]) if len(c) > 1 else "" for c in codes],
                        "reason_3": [REASON_KO.get(c[2], c[2]) if len(c) > 2 else "" for c in codes]})
    out = pd.concat([out, pts.reset_index(drop=True)], axis=1)
    from pathlib import Path
    o = Path(a.out); o.mkdir(parents=True, exist_ok=True)
    out.to_csv(o / "scores.csv", index=False)
    s_ok = score[ok]
    print(json.dumps({"rows": len(out), "unscorable_missing": int(bad.sum()), "mean_score": round(float(s_ok.mean()), 1) if ok.any() else None,
                      "below_cut_share": round(float((s_ok < ref["cut_score"]).mean()), 4) if ok.any() else None, "cut_score": ref["cut_score"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
