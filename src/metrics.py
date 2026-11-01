"""모형 평가 지표: AUC · Gini · KS · Brier · PSI · 점수 구간표."""
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

PSI_EPS = 1e-4


def ks_stat(y: np.ndarray, p: np.ndarray) -> float:
    """연체자와 정상 고객의 누적 분포 차이의 최댓값. p 는 연체 확률(클수록 위험)."""
    order = np.argsort(-p, kind="mergesort")
    ys = np.asarray(y)[order]
    cum_bad = np.cumsum(ys) / ys.sum()
    cum_good = np.cumsum(1 - ys) / (len(ys) - ys.sum())
    # 같은 점수끼리는 한 번에 넘어가야 하므로, 점수가 바뀌는 지점에서만 본다
    ps = np.asarray(p)[order]
    last_of_tie = np.r_[ps[1:] != ps[:-1], True]
    return float(np.max(np.abs(cum_bad - cum_good)[last_of_tie]))


def summary(y: np.ndarray, p: np.ndarray) -> dict:
    """p 가 확률이면 Brier(보정 정도)까지, 점수처럼 0~1 밖이면 순위 지표만."""
    auc = roc_auc_score(y, p)
    out = {"auc": round(auc, 4), "gini": round(2 * auc - 1, 4), "ks": round(ks_stat(y, p), 4)}
    if np.min(p) >= 0 and np.max(p) <= 1:
        out["brier"] = round(brier_score_loss(y, p), 4)
    return out


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """기준 집단(expected)의 분위 경계로 나눠 두 분포의 차이를 잰다.
    관례: 0.1 미만 안정 · 0.1~0.25 관찰 · 0.25 이상 재검토."""
    cuts = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)[1:-1]))
    e = np.bincount(np.searchsorted(cuts, expected, side="left"), minlength=len(cuts) + 1) / len(expected)
    a = np.bincount(np.searchsorted(cuts, actual, side="left"), minlength=len(cuts) + 1) / len(actual)
    e, a = np.clip(e, PSI_EPS, None), np.clip(a, PSI_EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def band_table(y: np.ndarray, score: np.ndarray, n_bands: int = 10) -> pd.DataFrame:
    """점수를 인원이 같은 구간으로 나눠 구간별 연체율. 1 구간 = 점수가 가장 낮은(위험한) 쪽."""
    df = pd.DataFrame({"y": y, "score": score})
    df["band"] = pd.qcut(df["score"].rank(method="first"), n_bands, labels=range(1, n_bands + 1))
    t = df.groupby("band", observed=True).agg(n=("y", "size"), bad=("y", "sum"),
                                              score_min=("score", "min"), score_max=("score", "max"))
    t["bad_rate"] = (t["bad"] / t["n"]).round(4)
    t["cum_bad_share"] = (t["bad"].cumsum() / t["bad"].sum()).round(4)
    return t.reset_index()
