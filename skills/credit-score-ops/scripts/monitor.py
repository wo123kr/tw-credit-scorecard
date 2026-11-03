"""운영 점검: 점수 · 피처 분포 이동(PSI), 결측 급증, 기준선 미만 비중, (라벨이 있으면) 구간별 연체율 · AUC · KS,
(보호 속성이 있으면) 집단별 보정 점검. 경고는 references/thresholds.json 기준. 숫자는 전부 여기서 계산하고 글은 보고서 틀에 끼운다.
uv run python skills/credit-score-ops/scripts/monitor.py --batch ops/batches/applicants_2026-02.csv --scores ops/runs/2026-02/scores.csv --out ops/runs/2026-02
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from _common import PROTECTED, REASON_KO, load_reference, thresholds
from metrics import psi, summary


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--batch", required=True); ap.add_argument("--scores", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(); th = thresholds(); ref = load_reference(); card = ref["card"]
    df = pd.read_csv(a.batch); sc = pd.read_csv(a.scores)
    assert (df["id"].to_numpy() == sc["id"].to_numpy()).all(), "배치와 점수 파일의 순서가 다름"
    alerts, notes = [], []
    r = {"batch": Path(a.batch).name, "rows": int(len(df))}
    # 0) 점수 불가 건(학습 때 없던 결측) — 있으면 먼저 알리고, 나머지 점검은 점수가 있는 행으로만
    unsc = sc["score"].isna().to_numpy(); r["unscorable"] = int(unsc.sum())
    if unsc.any():
        alerts.append(f"점수 불가 {int(unsc.sum()):,}건({unsc.mean():.1%}): 학습 때 결측이 없던 피처에 결측이 들어옴. 수집 · 적재를 먼저 확인")
    df, sc = df[~unsc].reset_index(drop=True), sc[~unsc].reset_index(drop=True)
    # 1) 점수 분포 이동
    r["psi_score"] = round(psi(ref["train_scores"], sc["score"].to_numpy()), 4)
    level = "안정" if r["psi_score"] < th["psi_watch"] else ("관찰" if r["psi_score"] < th["psi_review"] else "재검토")
    r["psi_score_level"] = level
    if level != "안정":
        alerts.append(f"점수 분포 이동 PSI {r['psi_score']} ({level}). 피처별 PSI 로 원인 피처를 확인")
    # 2) 피처별 PSI · 결측
    feat = {}
    for f in card.features:
        e = ref["train_features"][f].fillna(-1).to_numpy(); x = df[f].fillna(-1).to_numpy()
        m_ref, m_now = ref["train_missing_rate"][f], float(df[f].isna().mean())
        feat[f] = {"psi": round(psi(e, x), 4), "missing_ref": round(m_ref, 4), "missing_now": round(m_now, 4)}
        if feat[f]["psi"] >= th["psi_watch"]:
            alerts.append(f"피처 「{REASON_KO.get(f, f)}」 분포 이동 PSI {feat[f]['psi']}")
        if m_now - m_ref >= th["missing_rate_jump"]:
            alerts.append(f"피처 「{REASON_KO.get(f, f)}」 결측 {m_ref:.1%} → {m_now:.1%}. 수집 · 적재 확인 전엔 점수를 쓰지 말 것")
    r["features"] = feat
    # 3) 기준선 미만 비중
    below = float((sc["score"] < ref["cut_score"]).mean()); r["below_cut_share"] = round(below, 4)
    if abs(below - 0.20) > th["below_cut_share_tolerance"]:
        alerts.append(f"기준선 미만 비중 {below:.1%} (학습 기준 20%). 신청자 구성이 바뀌었는지 확인")
    # 4) 사유 코드 상위
    r["reason_top"] = sc["reason_1"].value_counts().head(5).to_dict()
    # 5) 라벨이 있으면 성과 · 구간별 연체율
    if "is_default" in df.columns and df["is_default"].notna().all():
        y = df["is_default"].to_numpy(); s = summary(y, -sc["score"].to_numpy())
        r["performance"] = {"auc": s["auc"], "ks": s["ks"], "bad_rate": round(float(y.mean()), 4)}
        if 0.757 - s["auc"] >= th["auc_drop_review"]:
            alerts.append(f"AUC {s['auc']} (검증 0.757 대비 {0.757 - s['auc']:.3f} 하락). 재학습 검토")
        tb = ref["train_bands"].set_index("band")["bad_rate"]
        g = pd.DataFrame({"band": sc["band"], "y": y}).groupby("band")["y"].agg(["size", "mean"])
        r["bands"] = [{"band": int(b), "n": int(row["size"]), "bad_rate": round(float(row["mean"]), 4), "bad_rate_train": float(tb.get(b, np.nan))} for b, row in g.iterrows()]
        notes.append("구간별 연체율은 학습 데이터 구간과 나란히 둠. 구간마다 표본이 작으면 흔들림")
    else:
        notes.append("라벨이 없어 성과(AUC · 구간별 연체율)는 계산하지 않음. 라벨이 쌓이면 다시 실행")
    # 6) 보호 속성이 있으면 집단별 보정 점검(모형에는 쓰지 않음)
    if all(c in df.columns for c in PROTECTED) and "is_default" in df.columns:
        d = df.assign(score=sc["score"].to_numpy(), p=sc["prob_default"].to_numpy(), ok=sc["score"].to_numpy() >= ref["cut_score"])
        d["age_band"] = pd.cut(d["age"], [0, 29, 44, 200], labels=["~29", "30~44", "45~"]); d["sex_ko"] = d["sex"].map({1: "남", 2: "여"})
        grp = {}
        for col in ["sex_ko", "age_band"]:
            for k, v in d.groupby(col, observed=True):
                ok = v[v["ok"]]
                gap = (float(ok["is_default"].mean()) - float(ok["p"].mean())) * 100 if len(ok) else float("nan")
                grp[f"{col}:{k}"] = {"n": int(len(v)), "approve_rate": round(float(v["ok"].mean()), 4),
                                     "bad_rate_approved": round(float(ok["is_default"].mean()), 4) if len(ok) else None,
                                     "pred_bad_approved": round(float(ok["p"].mean()), 4) if len(ok) else None,
                                     "calibration_gap_pp": round(gap, 2)}
                if len(v) >= th["min_group_n"] and abs(gap) >= th["calibration_gap_pp"]:
                    alerts.append(f"집단 {col}={k} 승인자 실제 연체율이 예측보다 {gap:+.1f}%p. 집단별 보정 점검 필요")
        r["groups"] = grp
        notes.append("성별 · 나이는 모형 피처가 아니며 사후 점검에만 씀")
    r["alerts"] = alerts; r["notes"] = notes
    o = Path(a.out); o.mkdir(parents=True, exist_ok=True)
    (o / "monitor.json").write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
    (o / "report.md").write_text(render(r), encoding="utf-8")
    print(json.dumps({"psi_score": r["psi_score"], "level": level, "alerts": alerts}, ensure_ascii=False))


def render(r: dict) -> str:
    L = [f"# 신용점수 운영 점검: {r['batch']}", "", f"- 신청 건수 {r['rows']:,}(점수 불가 {r.get('unscorable', 0):,}) · 점수 PSI {r['psi_score']} ({r['psi_score_level']}) · 기준선 미만 {r['below_cut_share']:.1%}"]
    if "performance" in r:
        p = r["performance"]; L.append(f"- 라벨 기준 성과: AUC {p['auc']} · KS {p['ks']} · 연체율 {p['bad_rate']:.1%}")
    L += ["", "## 경고"] + ([f"- {x}" for x in r["alerts"]] or ["- 없음"])
    L += ["", "## 피처별 분포 · 결측", "| 피처 | PSI | 결측(기준) | 결측(이번) |", "|---|---|---|---|"]
    L += [f"| {REASON_KO.get(f, f)} | {v['psi']} | {v['missing_ref']:.1%} | {v['missing_now']:.1%} |" for f, v in r["features"].items()]
    L += ["", "## 거절 사유 1순위 상위"] + [f"- {k}: {v:,}" for k, v in r["reason_top"].items()]
    if "bands" in r:
        L += ["", "## 점수 10구간 연체율(1 = 가장 낮은 점수)", "| 구간 | n | 이번 연체율 | 학습 연체율 |", "|---|---|---|---|"]
        L += [f"| {b['band']} | {b['n']} | {b['bad_rate']:.1%} | {b['bad_rate_train']:.1%} |" for b in r["bands"]]
    if "groups" in r:
        L += ["", "## 집단별 점검(승인자: 실제 연체율 − 예측 연체율)", "| 집단 | n | 승인율 | 실제 | 예측 | 차이(%p) |", "|---|---|---|---|---|---|"]
        L += [f"| {k} | {v['n']} | {v['approve_rate']:.1%} | {v['bad_rate_approved']:.1%} | {v['pred_bad_approved']:.1%} | {v['calibration_gap_pp']:+.1f} |" for k, v in r["groups"].items() if v["bad_rate_approved"] is not None]
    L += ["", "## 참고"] + [f"- {n}" for n in r["notes"]] + ["- 이 보고서는 점검 결과이며 승인 · 거절 결정은 심사 담당자가 한다"]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
