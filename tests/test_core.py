"""uv run python -m unittest discover -s tests"""
import sys
import unittest
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from card import BASE_ODDS, BASE_SCORE, PDO, fit_scorecard, reason_codes  # noqa: E402
from data import SQL  # noqa: E402
from metrics import ks_stat, psi  # noqa: E402
from woe import fit_binning  # noqa: E402


class MetricsTest(unittest.TestCase):
    def test_ks_hand_computed(self):
        # 위험 순: 0.9(1) 0.8(1) 0.7(0) 0.6(1) 0.5(0) 0.4(0)
        # 누적 연체 1/3 2/3 2/3 1 1 1, 누적 정상 0 0 1/3 1/3 2/3 1 → 최대 차 2/3
        y = np.array([1, 1, 0, 1, 0, 0])
        p = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
        self.assertAlmostEqual(ks_stat(y, p), 2 / 3)

    def test_ks_ties_counted_together(self):
        # 같은 점수의 연체 1 · 정상 1 은 한 번에 넘어가야 한다(정렬 순서로 KS 가 부풀지 않게)
        self.assertAlmostEqual(ks_stat(np.array([1, 0]), np.array([0.5, 0.5])), 0.0)

    def test_psi_same_is_zero_and_shift_is_large(self):
        rng = np.random.default_rng(0)
        a = rng.normal(size=20000)
        self.assertLess(psi(a, rng.normal(size=20000)), 0.01)
        self.assertGreater(psi(a, rng.normal(loc=1.0, size=20000)), 0.25)


class BinningTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.x = pd.Series(rng.uniform(0, 1, 5000))
        self.y = pd.Series((rng.uniform(size=5000) < 0.1 + 0.5 * self.x).astype(int))

    def test_bad_rate_monotonic_and_bins_big_enough(self):
        b = fit_binning("x", self.x, self.y)
        self.assertTrue(all(np.diff(b.bad_rate) >= 0))
        self.assertTrue(all(s >= 0.05 for s in b.share))
        self.assertGreater(b.iv, 0.1)

    def test_missing_gets_its_own_woe(self):
        x = self.x.copy()
        x.iloc[:300] = np.nan
        y = self.y.copy()
        y.iloc[:300] = 1   # 결측은 모두 연체 → 가장 위험한 WoE
        b = fit_binning("x", x, y)
        self.assertIsNotNone(b.missing_woe)
        self.assertGreater(b.missing_woe, max(b.woe))
        self.assertEqual(b.transform(x)[0], b.missing_woe)


class ScorecardTest(unittest.TestCase):
    def test_points_match_log_odds(self):
        rng = np.random.default_rng(2)
        n = 6000
        df = pd.DataFrame({"a": rng.uniform(size=n), "b": rng.integers(0, 4, n).astype(float)})
        logit = -2 + 2.5 * df["a"] + 0.6 * df["b"]
        y = pd.Series((rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int))
        card = fit_scorecard(df, y, ["a", "b"])
        p = card.predict_bad(df)
        factor = PDO / np.log(2)
        expected = BASE_SCORE - factor * np.log(BASE_ODDS) + factor * np.log((1 - p) / p)
        # 점수는 피처별로 반올림하므로 피처 수만큼(±0.5점씩) 차이가 날 수 있다
        self.assertLessEqual(np.max(np.abs(card.score(df) - expected)), 0.5 * len(card.features) + 1e-9)
        # 위험한 사람의 첫 사유는 점수를 가장 많이 깎은 피처
        worst = df.assign(s=card.score(df)).nsmallest(1, "s")
        self.assertTrue(reason_codes(card, worst)[0])


class FeatureSqlTest(unittest.TestCase):
    def _one(self, **over):
        row = {"id": 1, "is_default": 0, "sex": 1, "age": 30, "limit_bal": 1000.0, "education": 2,
               **{f"pay_{k}": 0 for k in (0, 2, 3, 4, 5, 6)},
               **{f"bill_amt{k}": 100.0 for k in range(1, 7)}, **{f"pay_amt{k}": 50.0 for k in range(1, 7)}}
        row.update(over)
        con = duckdb.connect()
        con.register("v", pd.DataFrame([row]))
        con.execute("CREATE TABLE raw_clients AS SELECT * FROM v")
        con.execute(SQL.read_text(encoding="utf-8"))
        return con.execute("SELECT * FROM features").df().iloc[0]

    def test_pay_ratio_null_when_no_previous_bill(self):
        # 직전 청구 0원이면 납부 비율은 「모름」이어야 한다(1.0 = 전액 납부로 읽히면 안 됨)
        f = self._one(bill_amt2=0.0)
        self.assertTrue(pd.isna(f["pay_ratio_now"]))

    def test_delinquency_counts(self):
        f = self._one(pay_0=2, pay_2=1, pay_5=3)
        self.assertEqual(f["delinq_now"], 2)
        self.assertEqual(f["delinq_max_6m"], 3)
        self.assertEqual(f["delinq_months_6m"], 3)
        self.assertEqual(f["delinq_months_3m"], 2)
        self.assertAlmostEqual(f["pay_ratio_now"], 0.5)


if __name__ == "__main__":
    unittest.main()
