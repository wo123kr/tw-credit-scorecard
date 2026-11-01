"""전체 실행: 피처 → 스코어카드 · 부스팅 학습 → 검증 → reports/ 에 결과.

uv run python src/run.py
"""
import json
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split

from card import fit_scorecard, reason_codes
from data import ROOT, load_features
from metrics import band_table, psi, summary

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SEED = 42
APPROVE_SHARE = 0.80   # 점수 상위 80% 승인으로 가정한 운영 시나리오
REPORTS = ROOT / "reports"
# 성별 · 결혼 · 나이는 피처에서 뺀다(사후 점검만). id · 라벨도 제외.
NOT_FEATURES = {"id", "is_default", "sex", "age"}
REASON_KO = {
    "delinq_now": "이번 달 연체", "delinq_max_6m": "최근 6개월 최장 연체", "delinq_months_6m": "6개월 중 연체한 달 수",
    "delinq_months_3m": "3개월 중 연체한 달 수", "idle_months_6m": "카드 미사용 달 수", "util_now": "이번 달 한도 소진율",
    "util_avg_6m": "6개월 평균 한도 소진율", "util_max_6m": "6개월 최고 한도 소진율", "bill_trend_6m": "청구액 증가 추세",
    "pay_ratio_avg_5m": "청구 대비 납부 비율(5개월 평균)", "pay_ratio_now": "직전 청구 대비 납부 비율",
    "zero_pay_months_5m": "청구가 있는데 0원 낸 달 수", "paid_to_limit_6m": "한도 대비 납부액", "limit_bal": "카드 한도",
    "education_grp": "학력 구분",
}


def gbm() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(learning_rate=0.05, max_iter=500, max_leaf_nodes=15,
                                          min_samples_leaf=100, l2_regularization=1.0,
                                          early_stopping=True, validation_fraction=0.15,
                                          n_iter_no_change=30, random_state=SEED)


def cross_validate(X: pd.DataFrame, y: pd.Series, feats: list[str]) -> dict:
    """학습 데이터 안에서 5겹 교차 검증. 구간화부터 매 겹마다 다시 한다(정보 새는 것 방지)."""
    aucs = {"scorecard": [], "gbm": []}
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y):
        card = fit_scorecard(X.iloc[tr], y.iloc[tr], feats)
        aucs["scorecard"].append(summary(y.iloc[va].to_numpy(), card.predict_bad(X.iloc[va]))["auc"])
        g = gbm().fit(X.iloc[tr][feats], y.iloc[tr])
        aucs["gbm"].append(summary(y.iloc[va].to_numpy(), g.predict_proba(X.iloc[va][feats])[:, 1])["auc"])
    return {k: {"mean": round(float(np.mean(v)), 4), "std": round(float(np.std(v)), 4)} for k, v in aucs.items()}


def approval_view(y: np.ndarray, risk: np.ndarray, cut: float) -> dict:
    """위험도가 cut 이하인 사람을 승인한다고 볼 때 승인율과 승인자 연체율."""
    ok = risk <= cut
    return {"approve_rate": round(float(ok.mean()), 4), "bad_rate_approved": round(float(y[ok].mean()), 4),
            "bad_rate_rejected": round(float(y[~ok].mean()), 4)}


def fairness(test: pd.DataFrame, score: np.ndarray, cut: float) -> dict:
    df = test.assign(score=score, ok=score >= cut)
    df["age_band"] = pd.cut(df["age"], [0, 29, 44, 200], labels=["~29", "30~44", "45~"])
    df["sex_ko"] = df["sex"].map({1: "남", 2: "여"})
    out = {}
    for col in ["sex_ko", "age_band"]:
        g = df.groupby(col, observed=True)
        out[col] = {str(k): {"n": int(len(v)), "approve_rate": round(float(v["ok"].mean()), 4),
                             "bad_rate_all": round(float(v["is_default"].mean()), 4),
                             "bad_rate_approved": round(float(v.loc[v["ok"], "is_default"].mean()), 4)}
                    for k, v in g}
    return out


def plot(test_y: np.ndarray, score: np.ndarray, bands: pd.DataFrame) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].hist(score[test_y == 0], bins=40, alpha=0.6, label="정상(0)", density=True)
    ax[0].hist(score[test_y == 1], bins=40, alpha=0.6, label="연체(1)", density=True)
    ax[0].set_title("검증 데이터 점수 분포"), ax[0].set_xlabel("점수"), ax[0].legend()
    ax[1].bar(bands["band"].astype(int), bands["bad_rate"] * 100)
    ax[1].set_title("점수 10구간별 연체율(1 = 가장 낮은 점수)"), ax[1].set_xlabel("구간"), ax[1].set_ylabel("%")
    fig.tight_layout()
    fig.savefig(REPORTS / "score_bands.png", dpi=130)


def main() -> None:
    plt.rcParams["font.family"] = ["AppleGothic", "DejaVu Sans"]
    REPORTS.mkdir(exist_ok=True)
    df = load_features()
    feats = [c for c in df.columns if c not in NOT_FEATURES]
    train, test = train_test_split(df, test_size=0.3, stratify=df["is_default"], random_state=SEED)
    ytr, yte = train["is_default"], test["is_default"].to_numpy()

    card = fit_scorecard(train, ytr, feats)
    g = gbm().fit(train[feats], ytr)
    p_card, p_gbm = card.predict_bad(test), g.predict_proba(test[feats])[:, 1]
    s_tr, s_te = card.score(train), card.score(test)

    cut_score = float(np.quantile(s_tr, 1 - APPROVE_SHARE))   # 학습 데이터 기준 하위 20% 점수
    cut_gbm = float(np.quantile(g.predict_proba(train[feats])[:, 1], APPROVE_SHARE))
    bands = band_table(yte, s_te)
    rejected = test[s_te < cut_score]
    codes = reason_codes(card, rejected)
    first = Counter(REASON_KO[r[0]] for r in codes if r)
    any3 = Counter(REASON_KO[f] for r in codes for f in r)

    results = {
        "data": {"rows": len(df), "bad_rate": round(float(df["is_default"].mean()), 4),
                 "train": len(train), "test": len(test), "candidate_features": len(feats)},
        "scorecard_features": card.features,
        "dropped": card.dropped,
        "iv": {f: round(b.iv, 4) for f, b in sorted(card.binnings.items(), key=lambda kv: -kv[1].iv)},
        "test": {"baseline_delinq_now_only": summary(yte, test["delinq_now"].to_numpy()),
                 "scorecard_prob": summary(yte, p_card), "scorecard_points": summary(yte, -s_te),
                 "gbm": summary(yte, p_gbm)},
        "cv_train_5fold_auc": cross_validate(train.reset_index(drop=True), ytr.reset_index(drop=True), feats),
        "approve_80": {"scorecard": approval_view(yte, -s_te, -cut_score), "gbm": approval_view(yte, p_gbm, cut_gbm),
                       "cut_score": cut_score},
        "bands_monotonic": bool(bands["bad_rate"].is_monotonic_decreasing),
        "psi_score_train_vs_test": round(psi(s_tr, s_te), 4),
        "psi_features_max": round(max(psi(train[f].fillna(-1).to_numpy(), test[f].fillna(-1).to_numpy())
                                      for f in card.features), 4),
        "reasons_rejected": {"n": len(rejected), "first": dict(first.most_common()),
                             "in_top3": dict(any3.most_common())},
        "fairness_at_approve_80": fairness(test, s_te, cut_score),
    }
    (REPORTS / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    card.table.to_csv(REPORTS / "scorecard_table.csv", index=False)
    bands.to_csv(REPORTS / "score_bands.csv", index=False)
    plot(yte, s_te, bands)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
