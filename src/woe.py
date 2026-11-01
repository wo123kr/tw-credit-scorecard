"""WoE 구간화. 연체율이 한 방향으로만 움직이도록(단조) 이웃 구간을 합친다.

WoE = ln(연체자 분포 / 정상 분포). 값이 클수록 위험하다(로지스틱 계수가 양수로 나오게).
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

MIN_BIN_SHARE = 0.05   # 구간 하나가 전체의 5% 는 넘어야 한다
MAX_INIT_BINS = 10
SMOOTH = 0.5           # 빈 구간에서 ln(0) 이 나오지 않게 더하는 값


@dataclass
class Binning:
    feature: str
    cuts: list[float]          # 안쪽 경계. 구간 i = (cuts[i-1], cuts[i]]
    woe: list[float]           # 구간별 WoE
    bad_rate: list[float]
    share: list[float]
    missing_woe: float | None  # 결측 구간(없으면 None)
    iv: float

    def bin_index(self, x: pd.Series) -> np.ndarray:
        return np.searchsorted(np.asarray(self.cuts), x.to_numpy(dtype=float), side="left")

    def transform(self, x: pd.Series) -> np.ndarray:
        idx = self.bin_index(x)
        out = np.asarray(self.woe)[np.minimum(idx, len(self.woe) - 1)]
        miss = x.isna().to_numpy()
        out[miss] = self.missing_woe if self.missing_woe is not None else 0.0
        return out

    def labels(self) -> list[str]:
        edges = [-np.inf, *self.cuts, np.inf]
        return [f"({edges[i]:.4g}, {edges[i + 1]:.4g}]" for i in range(len(self.woe))]


def _initial_cuts(x: pd.Series) -> list[float]:
    vals = np.sort(x.unique())
    if len(vals) <= MAX_INIT_BINS:
        return [float(v) for v in vals[:-1]]
    qs = np.quantile(x, np.linspace(0, 1, MAX_INIT_BINS + 1)[1:-1])
    return sorted({float(q) for q in qs if q < vals[-1]})


def _stats(x: pd.Series, y: pd.Series, cuts: list[float]) -> pd.DataFrame:
    idx = np.searchsorted(np.asarray(cuts), x.to_numpy(dtype=float), side="left")
    g = pd.DataFrame({"b": idx, "y": y.to_numpy()}).groupby("b")["y"]
    return pd.DataFrame({"n": g.size(), "bad": g.sum()}).reindex(range(len(cuts) + 1), fill_value=0)


def _merge_once(st: pd.DataFrame, cuts: list[float], increasing: bool, n_total: int) -> list[float] | None:
    """규칙을 어기는 첫 이웃 쌍을 찾아 그 경계를 지운 새 경계를 돌려준다. 어긴 곳이 없으면 None."""
    rate = (st["bad"] / st["n"].clip(lower=1)).to_numpy()
    share = st["n"].to_numpy() / n_total
    for i in range(len(cuts)):
        small = share[i] < MIN_BIN_SHARE or share[i + 1] < MIN_BIN_SHARE
        wrong = rate[i + 1] < rate[i] if increasing else rate[i + 1] > rate[i]
        if small or wrong:
            return cuts[:i] + cuts[i + 1:]
    return None


def _woe_values(bad: np.ndarray, good: np.ndarray, tot_bad: float, tot_good: float) -> np.ndarray:
    return np.log(((bad + SMOOTH) / tot_bad) / ((good + SMOOTH) / tot_good))


def fit_binning(feature: str, x: pd.Series, y: pd.Series) -> Binning:
    miss = x.isna()
    xs, ys = x[~miss], y[~miss]
    increasing = np.corrcoef(xs.rank(), ys)[0, 1] >= 0
    cuts = _initial_cuts(xs)
    while (new := _merge_once(_stats(xs, ys, cuts), cuts, increasing, len(x))) is not None:
        cuts = new
    st = _stats(xs, ys, cuts)
    tot_bad, tot_good = float(y.sum()), float(len(y) - y.sum())
    bad, good = st["bad"].to_numpy(float), (st["n"] - st["bad"]).to_numpy(float)
    woe = _woe_values(bad, good, tot_bad, tot_good)
    iv = float(np.sum((bad / tot_bad - good / tot_good) * woe))
    missing_woe = None
    if miss.any():
        mb, mg = float(y[miss].sum()), float(miss.sum() - y[miss].sum())
        missing_woe = float(_woe_values(np.array([mb]), np.array([mg]), tot_bad, tot_good)[0])
        iv += (mb / tot_bad - mg / tot_good) * missing_woe
    return Binning(feature, cuts, woe.tolist(), (st["bad"] / st["n"].clip(lower=1)).tolist(),
                   (st["n"] / len(x)).tolist(), missing_woe, iv)
