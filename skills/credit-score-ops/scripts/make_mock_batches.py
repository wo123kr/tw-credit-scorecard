"""모의 신청 배치 3개 — 데이터가 한 시점뿐이라 검증 데이터(9,000명)에서 뽑아 「달」로 가정한다.
2026-01: 무작위(안정) · 2026-02: 고위험 쪽으로 치우친 표본(분포 이동) · 2026-03: 한도 값 일부 결측(데이터 품질).
라벨(is_default)은 「나중에 알게 된 결과」로 가정해 같이 둔다.
"""
import numpy as np

from _common import OPS, load_reference
from data import load_features

N = 3000


def main() -> None:
    out = OPS / "batches"; out.mkdir(parents=True, exist_ok=True)
    ref = load_reference()
    df = load_features()
    test = df[df["id"].isin(ref["test_ids"])].reset_index(drop=True)
    rng = np.random.default_rng(2026)
    a = test.sample(N, random_state=1)
    w = 1 + 3 * (test["delinq_now"] > 0) + 2 * (test["util_avg_6m"] > 0.8)
    b = test.sample(N, weights=w, random_state=2)
    c = test.sample(N, random_state=3).copy()
    c.loc[c.sample(frac=0.3, random_state=4).index, "limit_bal"] = np.nan
    for name, d in [("2026-01", a), ("2026-02", b), ("2026-03", c)]:
        d.to_csv(out / f"applicants_{name}.csv", index=False)
        print(name, len(d), "bad_rate", round(float(d["is_default"].mean()), 4), "delinq_now>0", round(float((d["delinq_now"] > 0).mean()), 3))


if __name__ == "__main__":
    main()
