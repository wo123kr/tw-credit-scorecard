"""운영 점검 스킬 시험 — 참조 · 모의 배치가 ops/ 에 있어야 한다(fit_reference · make_mock_batches 먼저)."""
import json
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
S = ROOT / "skills" / "credit-score-ops" / "scripts"
sys.path.insert(0, str(S)); sys.path.insert(0, str(ROOT / "src"))
from _common import load_reference  # noqa: E402


def run(script, *args):
    return subprocess.run([sys.executable, str(S / script), *args], capture_output=True, text=True, cwd=ROOT)


class Ops(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ref = load_reference()
        cls.out = ROOT / "ops" / "runs" / "_test"
        for m in ("2026-01", "2026-02", "2026-03"):
            b = ROOT / "ops" / "batches" / f"applicants_{m}.csv"
            r = run("score_batch.py", "--batch", str(b), "--out", str(cls.out / m)); assert r.returncode == 0, r.stderr
            r = run("monitor.py", "--batch", str(b), "--scores", str(cls.out / m / "scores.csv"), "--out", str(cls.out / m)); assert r.returncode == 0, r.stderr
        cls.mon = {m: json.loads((cls.out / m / "monitor.json").read_text(encoding="utf-8")) for m in ("2026-01", "2026-02", "2026-03")}

    def test_scores_match_card(self):
        """스킬이 낸 점수 = 원래 스코어카드 점수(같은 행)."""
        b = pd.read_csv(ROOT / "ops" / "batches" / "applicants_2026-01.csv"); sc = pd.read_csv(self.out / "2026-01" / "scores.csv")
        np.testing.assert_array_equal(sc["score"].to_numpy(), self.ref["card"].score(b))
        self.assertTrue((sc["band"].between(1, 10)).all())

    def test_stable_batch_no_psi_alert(self):
        m = self.mon["2026-01"]; self.assertLess(m["psi_score"], 0.1); self.assertEqual(m["psi_score_level"], "안정")
        self.assertFalse(any("분포 이동" in a for a in m["alerts"]))

    def test_shifted_batch_alerts(self):
        m = self.mon["2026-02"]; self.assertGreaterEqual(m["psi_score"], 0.1)
        self.assertTrue(any("점수 분포 이동" in a for a in m["alerts"]))
        self.assertTrue(any("기준선 미만 비중" in a for a in m["alerts"]))

    def test_missing_batch_unscorable(self):
        """학습 때 결측이 없던 피처(카드 한도)에 결측이 오면 그 행은 점수 불가, 경고가 뜬다."""
        sc = pd.read_csv(self.out / "2026-03" / "scores.csv"); m = self.mon["2026-03"]
        self.assertEqual(int(sc["score"].isna().sum()), m["unscorable"]); self.assertGreater(m["unscorable"], 0)
        self.assertTrue((sc.loc[sc["score"].isna(), "vs_cut"] == "점수 불가(결측)").all())
        self.assertTrue(any("점수 불가" in a for a in m["alerts"]))

    def test_no_decision_words(self):
        for m in self.mon.values():
            txt = (self.out / m["batch"].replace("applicants_", "").replace(".csv", "") / "report.md").read_text(encoding="utf-8")
            self.assertNotIn("승인함", txt); self.assertNotIn("거절함", txt); self.assertIn("심사 담당자", txt)

    def test_explain_reasons(self):
        sc = pd.read_csv(self.out / "2026-01" / "scores.csv"); i = int(sc[sc["vs_cut"] == "기준선 미만"]["id"].iloc[0])
        r = run("explain.py", "--scores", str(self.out / "2026-01" / "scores.csv"), "--id", str(i)); self.assertEqual(r.returncode, 0, r.stderr)
        j = json.loads(r.stdout); self.assertEqual(len([x for x in j["reasons"] if x]), 3)
        self.assertEqual(j["points"][0]["gap"], max(p["gap"] for p in j["points"]))


if __name__ == "__main__":
    unittest.main()
