"""검증 데이터 AUC · KS 의 부트스트랩 95% 구간(1,000회). 모형 선택은 바꾸지 않는 사후 계산.
uv run python src/ci.py → reports/ci.json
"""
import json

import numpy as np
from sklearn.model_selection import train_test_split

from card import fit_scorecard
from data import ROOT, load_features
from metrics import summary
from run import NOT_FEATURES, SEED, gbm

B = 1000


def boot(y, p, rng):
    aucs, kss = [], []
    n = len(y)
    for _ in range(B):
        i = rng.integers(0, n, n); s = summary(y[i], p[i]); aucs.append(s["auc"]); kss.append(s["ks"])
    q = lambda v: [round(float(np.quantile(v, 0.025)), 4), round(float(np.quantile(v, 0.975)), 4)]
    return {"auc_ci95": q(aucs), "ks_ci95": q(kss)}


def main() -> None:
    df = load_features(); feats = [c for c in df.columns if c not in NOT_FEATURES]
    train, test = train_test_split(df, test_size=0.3, stratify=df["is_default"], random_state=SEED)
    card = fit_scorecard(train, train["is_default"], feats); g = gbm().fit(train[feats], train["is_default"])
    yte = test["is_default"].to_numpy(); rng = np.random.default_rng(SEED)
    models = {"baseline_delinq_now_only": test["delinq_now"].to_numpy().astype(float), "scorecard": card.predict_bad(test), "gbm": g.predict_proba(test[feats])[:, 1]}
    out = {"n_test": int(len(test)), "bootstrap": B, "split": "70:30 층화 분할(seed 42), 구간화 · 학습은 학습 데이터에서만, 교차 검증은 학습 데이터 안 5겹"}
    for k, p in models.items():
        out[k] = {**summary(yte, p), **boot(yte, p, rng)}
    # 스코어카드 - 부스팅 AUC 차이의 구간(같은 표본으로)
    i_all = [rng.integers(0, len(yte), len(yte)) for _ in range(B)]
    diff = [summary(yte[i], models["gbm"][i])["auc"] - summary(yte[i], models["scorecard"][i])["auc"] for i in i_all]
    out["gbm_minus_scorecard_auc_ci95"] = [round(float(np.quantile(diff, 0.025)), 4), round(float(np.quantile(diff, 0.975)), 4)]
    (ROOT / "reports" / "ci.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
