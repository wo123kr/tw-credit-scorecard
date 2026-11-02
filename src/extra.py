"""추가 분석: run.py 와 같은 분할 · 같은 모형 설정으로, 모형은 바꾸지 않고 면접에서 나올 질문을 숫자로 확인한다.

1. 부스팅이 앞선 0.025 는 모형 덕인가 피처 수 덕인가 — 같은 피처로 맞바꿔 본다(학습 데이터 5겹 교차 검증)
2. 단조 제약을 건 부스팅 — 설명 가능성을 지키면서 얼마나 남나
3. 확률 보정 — 예측 연체율과 실제 연체율(점수 10구간)
4. 같은 승인율에서 두 모형이 엇갈리게 승인하는 사람(swap set)과 승인율별 연체율
5. 집단별 보정 — 같은 점수 구간 안에서 성별 · 나이대 실제 연체율, 승인자의 예측 대비 실제

uv run python src/extra.py   → reports/extra.json
"""
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split

from card import fit_scorecard
from data import load_features
from metrics import summary
from run import APPROVE_SHARE, NOT_FEATURES, REPORTS, SEED, gbm
from woe import fit_binning


def gbm_mono(directions: list[int]) -> HistGradientBoostingClassifier:
    g = gbm()
    g.set_params(monotonic_cst=directions)
    return g


UNORDERED = {"education_grp"}   # 1~3 은 학력 순서지만 4 는 「기타」 묶음이라 순서가 없다 → 제약 안 걸음


def directions(train: pd.DataFrame, y: pd.Series, feats: list[str]) -> list[int]:
    """학습 데이터에서 피처와 연체의 순위 상관 부호 = 단조 제약 방향(+1 이면 값이 클수록 위험, 0 은 제약 없음)."""
    out = []
    for f in feats:
        if f in UNORDERED:
            out.append(0)
            continue
        r = pd.Series(train[f]).rank().corr(y.rank())
        out.append(1 if r > 0 else -1)
    return out


def woe_logit_all(train: pd.DataFrame, y: pd.Series, feats: list[str]):
    """피처 고르기 없이 15개 전부 WoE 로 바꿔 로지스틱 — 스코어카드 방식에 피처만 늘린 것."""
    bins = {f: fit_binning(f, train[f], y) for f in feats}

    def woe(df):
        return pd.DataFrame({f: bins[f].transform(df[f]) for f in feats}, index=df.index)

    m = LogisticRegression(C=1.0, max_iter=2000).fit(woe(train), y)
    return lambda df: m.predict_proba(woe(df))[:, 1]


def cv_auc(train: pd.DataFrame, y: pd.Series, fit_predict) -> dict:
    aucs = []
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=SEED).split(train, y):
        p = fit_predict(train.iloc[tr], y.iloc[tr], train.iloc[va])
        aucs.append(summary(y.iloc[va].to_numpy(), p)["auc"])
    return {"mean": round(float(np.mean(aucs)), 4), "std": round(float(np.std(aucs)), 4)}


def approve_top(risk: np.ndarray, share: float) -> np.ndarray:
    """위험이 낮은 순서로 share 만큼 승인(동점은 순서대로 끊음)."""
    order = np.argsort(risk, kind="mergesort")
    ok = np.zeros(len(risk), dtype=bool)
    ok[order[: int(round(share * len(risk)))]] = True
    return ok


def main() -> None:
    df = load_features()
    feats = [c for c in df.columns if c not in NOT_FEATURES]
    train, test = train_test_split(df, test_size=0.3, stratify=df["is_default"], random_state=SEED)
    train, test = train.reset_index(drop=True), test.reset_index(drop=True)
    ytr, yte = train["is_default"], test["is_default"].to_numpy()

    card = fit_scorecard(train, ytr, feats)
    sc8 = card.features

    # ── 1 · 2. 모형 × 피처 맞바꾸기 (학습 데이터 5겹 + 검증 데이터 한 번)
    def fp_card(tr, y, va):
        return fit_scorecard(tr, y, feats).predict_bad(va)

    variants = {
        "scorecard_8 (기존)": fp_card,
        "woe_logistic_15": lambda tr, y, va: woe_logit_all(tr, y, feats)(va),
        "gbm_8 (스코어카드와 같은 피처)": lambda tr, y, va: gbm().fit(tr[sc8], y).predict_proba(va[sc8])[:, 1],
        "gbm_15 (기존)": lambda tr, y, va: gbm().fit(tr[feats], y).predict_proba(va[feats])[:, 1],
        # 단조 방향도 겹마다 그 겹의 학습 부분에서 정한다(검증 겹 정보가 섞이지 않게)
        "gbm_15_monotonic": lambda tr, y, va: gbm_mono(directions(tr, y, feats)).fit(tr[feats], y).predict_proba(va[feats])[:, 1],
        "gbm_8_monotonic": lambda tr, y, va: gbm_mono(directions(tr, y, sc8)).fit(tr[sc8], y).predict_proba(va[sc8])[:, 1],
    }
    ablation = {}
    preds = {}
    for name, fp in variants.items():
        p = fp(train, ytr, test)
        preds[name] = p
        ablation[name] = {"cv5_auc": cv_auc(train, ytr, fp), "test": summary(yte, p)}

    # ── 3. 확률 보정 (스코어카드 확률, 검증 데이터 10구간)
    p_card = preds["scorecard_8 (기존)"]
    p_gbm = preds["gbm_15 (기존)"]
    cal = pd.DataFrame({"p": p_card, "y": yte})
    cal["decile"] = pd.qcut(cal["p"].rank(method="first"), 10, labels=range(1, 11))
    cal_t = cal.groupby("decile", observed=True).agg(pred=("p", "mean"), actual=("y", "mean"), n=("y", "size"))
    calibration = {
        "scorecard_deciles": [{"decile": int(k), "pred": round(float(r.pred), 4), "actual": round(float(r.actual), 4)}
                              for k, r in cal_t.iterrows()],
        "scorecard_mean_pred_vs_actual": [round(float(p_card.mean()), 4), round(float(yte.mean()), 4)],
        "max_abs_gap_pp": round(float((cal_t["pred"] - cal_t["actual"]).abs().max() * 100), 1),
    }

    # ── 4. 승인율별 연체율 · swap set
    curve = {}
    for share in (0.5, 0.6, 0.7, 0.8, 0.9):
        a, b = approve_top(p_card, share), approve_top(p_gbm, share)
        curve[f"{int(share * 100)}%"] = {"scorecard": round(float(yte[a].mean()), 4),
                                         "gbm": round(float(yte[b].mean()), 4)}
    a, b = approve_top(p_card, APPROVE_SHARE), approve_top(p_gbm, APPROVE_SHARE)
    swap = {
        "approve_share": APPROVE_SHARE,
        "both_approve": {"n": int((a & b).sum()), "bad_rate": round(float(yte[a & b].mean()), 4)},
        "card_only (스코어카드만 승인)": {"n": int((a & ~b).sum()), "bad_rate": round(float(yte[a & ~b].mean()), 4)},
        "gbm_only (부스팅만 승인)": {"n": int((~a & b).sum()), "bad_rate": round(float(yte[~a & b].mean()), 4)},
        "both_reject": {"n": int((~a & ~b).sum()), "bad_rate": round(float(yte[~a & ~b].mean()), 4)},
    }

    # ── 5. 집단별 보정: 같은 점수 구간 안에서의 실제 연체율, 승인자의 예측 대비 실제
    s_te = card.score(test)
    g = test.assign(p=p_card, s=s_te)
    g["sex_ko"] = g["sex"].map({1: "남", 2: "여"})
    g["age_band"] = pd.cut(g["age"], [0, 29, 44, 200], labels=["~29", "30~44", "45~"])
    g["band5"] = pd.qcut(g["s"].rank(method="first"), 5, labels=["1(낮음)", "2", "3", "4", "5(높음)"])
    # run.py 의 공정성 점검과 같은 승인 기준: 학습 데이터 점수 하위 20% 경계(검증 데이터 승인율은 80.3%)
    cut_score = float(np.quantile(card.score(train), 1 - APPROVE_SHARE))
    approved = s_te >= cut_score
    group_cal = {"cut_score": cut_score, "approve_rate": round(float(approved.mean()), 4)}
    for col in ("sex_ko", "age_band"):
        by_band = (g.groupby(["band5", col], observed=True)["is_default"].mean().unstack().round(4))
        appr = g[approved].groupby(col, observed=True).agg(n=("p", "size"), pred=("p", "mean"),
                                                            actual=("is_default", "mean"))
        group_cal[col] = {
            "bad_rate_within_score_quintile": {str(k): {str(c): float(v) for c, v in row.items()}
                                               for k, row in by_band.iterrows()},
            "approved_pred_vs_actual": {str(k): {"n": int(r.n), "pred": round(float(r.pred), 4),
                                                 "actual": round(float(r.actual), 4),
                                                 "gap_pp": round(float((r.actual - r.pred) * 100), 1)}
                                        for k, r in appr.iterrows()},
        }
        # 나이대 · 성별 차이가 점수 구간 차이 때문인지 보려고, 승인자 안 평균 점수도 남긴다
        group_cal[col]["approved_mean_score"] = {str(k): round(float(v), 1)
                                                 for k, v in g[approved].groupby(col, observed=True)["s"].mean().items()}

    out = {"note": "run.py 와 같은 분할(70:30, seed 42) · 같은 설정. 모형 선택은 바꾸지 않은 사후 분석.",
           "monotonic_directions": dict(zip(feats, directions(train, ytr, feats))),
           "ablation": ablation, "calibration": calibration,
           "approval_curve_bad_rate": curve, "swap_set_at_80": swap, "group_calibration": group_cal}
    (REPORTS / "extra.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
