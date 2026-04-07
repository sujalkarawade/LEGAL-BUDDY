import json
from functools import lru_cache
from pathlib import Path

import pandas as pd

from app.config import RISK_PATH


@lru_cache(maxsize=4)
def load_json_list(path: Path, key: str | None = None):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if key:
        return data.get(key, [])
    return data


@lru_cache(maxsize=1)
def load_risk_data() -> pd.DataFrame:
    if not RISK_PATH.exists():
        return pd.DataFrame(columns=["clause_type", "risk_level"])
    return pd.read_csv(RISK_PATH)
