"""원본 엑셀 → DuckDB raw_clients → features.sql 로 피처 테이블."""
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_XLS = ROOT / "data" / "raw" / "default of credit card clients.xls"
SQL = Path(__file__).with_name("features.sql")

# 원본 열 이름 → 소문자. 라벨 열 이름이 길어 is_default 로 바꾼다.
LABEL_SRC = "default payment next month"


def load_raw(path: Path = RAW_XLS) -> pd.DataFrame:
    df = pd.read_excel(path, header=1)
    df = df.rename(columns={LABEL_SRC: "is_default"})
    df.columns = [c.lower() for c in df.columns]
    if len(df) != 30000 or df["id"].duplicated().any():
        raise ValueError(f"원본 행 수가 예상과 다름: {len(df)}")
    return df


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    con = duckdb.connect()
    con.register("raw_view", raw)
    con.execute("CREATE TABLE raw_clients AS SELECT * FROM raw_view")
    con.execute(SQL.read_text(encoding="utf-8"))
    out = con.execute("SELECT * FROM features ORDER BY id").df()
    con.close()
    return out


def load_features() -> pd.DataFrame:
    return build_features(load_raw())
