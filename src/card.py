"""WoE 로지스틱 스코어카드: 피처 고르기 → 학습 → 점수표(PDO) → 거절 사유 코드."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from woe import Binning, fit_binning

IV_MIN = 0.02          # 정보가치가 이보다 작으면 쓰지 않는다
CORR_MAX = 0.8         # WoE 끼리 이보다 비슷하면 IV 가 큰 쪽만 남긴다
PDO = 20               # 점수 20점 = 정상:연체 오즈 두 배
BASE_SCORE = 600
BASE_ODDS = 50         # 600점 = 정상:연체 50:1


@dataclass
class Scorecard:
    binnings: dict[str, Binning]
    features: list[str]
    model: LogisticRegression
    table: pd.DataFrame        # feature · bin · woe · points
    dropped: dict[str, str]    # 뺀 피처와 이유

    def woe_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({f: self.binnings[f].transform(df[f]) for f in self.features}, index=df.index)

    def predict_bad(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self.woe_frame(df))[:, 1]

    def points(self, df: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for f in self.features:
            t = self.table[self.table["feature"] == f]
            b = self.binnings[f]
            idx = np.minimum(b.bin_index(df[f]), len(b.woe) - 1)
            pts = t[t["bin"] != "missing"]["points"].to_numpy()[idx].astype(float)
            miss = df[f].isna().to_numpy()
            if miss.any():
                pts[miss] = t[t["bin"] == "missing"]["points"].iloc[0]
            out[f] = pts
        return pd.DataFrame(out, index=df.index)

    def score(self, df: pd.DataFrame) -> np.ndarray:
        return self.points(df).sum(axis=1).to_numpy()


def _select(binnings: dict[str, Binning], woe: pd.DataFrame) -> tuple[list[str], dict[str, str]]:
    dropped = {f: f"IV {b.iv:.3f} < {IV_MIN}" for f, b in binnings.items() if b.iv < IV_MIN}
    keep = sorted((f for f in binnings if f not in dropped), key=lambda f: -binnings[f].iv)
    chosen: list[str] = []
    corr = woe.corr()
    for f in keep:
        clash = next((c for c in chosen if abs(corr.loc[f, c]) > CORR_MAX), None)
        if clash:
            dropped[f] = f"{clash} 와 상관 {corr.loc[f, clash]:.2f}"
        else:
            chosen.append(f)
    return chosen, dropped


def _fit_signed(woe: pd.DataFrame, y: pd.Series, feats: list[str], dropped: dict[str, str]):
    """WoE 는 클수록 위험하므로 계수는 양수여야 한다. 음수가 나오면(다른 피처와 겹친 것) 빼고 다시."""
    while True:
        m = LogisticRegression(C=1.0, max_iter=2000).fit(woe[feats], y)
        neg = [f for f, c in zip(feats, m.coef_[0]) if c <= 0]
        if not neg:
            return m, feats
        worst = min(neg, key=lambda f: m.coef_[0][feats.index(f)])
        dropped[worst] = "계수 부호 반대(다른 피처와 정보 겹침)"
        feats = [f for f in feats if f != worst]


def _points_table(binnings: dict[str, Binning], feats: list[str], m: LogisticRegression) -> pd.DataFrame:
    factor = PDO / np.log(2)
    offset = BASE_SCORE - factor * np.log(BASE_ODDS)
    n, b0 = len(feats), m.intercept_[0]
    rows = []
    for f, coef in zip(feats, m.coef_[0]):
        b = binnings[f]
        pairs = list(zip(b.labels(), b.woe, b.bad_rate, b.share))
        if b.missing_woe is not None:
            pairs.append(("missing", b.missing_woe, None, None))
        for label, w, rate, share in pairs:
            pts = round(offset / n - factor * (coef * w + b0 / n))
            rows.append({"feature": f, "bin": label, "woe": round(w, 4), "bad_rate": rate,
                         "share": share, "points": pts})
    return pd.DataFrame(rows)


def fit_scorecard(df: pd.DataFrame, y: pd.Series, candidates: list[str]) -> Scorecard:
    binnings = {f: fit_binning(f, df[f], y) for f in candidates}
    woe = pd.DataFrame({f: b.transform(df[f]) for f, b in binnings.items()}, index=df.index)
    feats, dropped = _select(binnings, woe)
    m, feats = _fit_signed(woe, y, feats, dropped)
    return Scorecard(binnings, feats, m, _points_table(binnings, feats, m), dropped)


def reason_codes(card: Scorecard, df: pd.DataFrame, top: int = 3) -> list[list[str]]:
    """사람마다 「그 피처에서 받을 수 있는 최고 점수 - 받은 점수」가 큰 순으로 사유를 고른다."""
    pts = card.points(df)
    best = card.table.groupby("feature")["points"].max()
    gap = best[pts.columns].to_numpy() - pts.to_numpy()
    cols = np.asarray(pts.columns)
    out = []
    for row in gap:
        order = np.argsort(-row, kind="mergesort")[:top]
        out.append([str(cols[i]) for i in order if row[i] > 0])
    return out
