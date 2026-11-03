"""참조 만들기: 원래 분석과 같은 분할 · 시드로 스코어카드를 다시 학습해 ops/reference/ 에 저장.
uv run python skills/credit-score-ops/scripts/fit_reference.py
"""
import json
import pickle

import numpy as np
from sklearn.model_selection import train_test_split

from _common import REF_DIR
from card import fit_scorecard
from data import load_features
from metrics import band_table

SEED, APPROVE_SHARE = 42, 0.80
NOT_FEATURES = {"id", "is_default", "sex", "age"}


def main() -> None:
    REF_DIR.mkdir(parents=True, exist_ok=True)
    df = load_features()
    feats = [c for c in df.columns if c not in NOT_FEATURES]
    train, test = train_test_split(df, test_size=0.3, stratify=df["is_default"], random_state=SEED)
    card = fit_scorecard(train, train["is_default"], feats)
    s_tr = card.score(train)
    cut = float(np.quantile(s_tr, 1 - APPROVE_SHARE))
    bands = band_table(train["is_default"].to_numpy(), s_tr)
    ref = {"card": card, "cut_score": cut, "train_scores": s_tr, "train_prob": card.predict_bad(train),
           "train_features": train[card.features].reset_index(drop=True),
           "train_missing_rate": {f: float(train[f].isna().mean()) for f in card.features},
           "train_bands": bands, "test_ids": test["id"].to_numpy(), "train_bad_rate": float(train["is_default"].mean())}
    with open(REF_DIR / "reference.pkl", "wb") as f:
        pickle.dump(ref, f)
    meta = {"features": card.features, "cut_score": cut, "train_rows": len(train), "train_bad_rate": ref["train_bad_rate"],
            "band_edges": [float(x) for x in bands["score_min"]], "train_bands": bands.to_dict(orient="records")}
    (REF_DIR / "reference.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"features": card.features, "cut_score": cut, "train_rows": len(train)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
