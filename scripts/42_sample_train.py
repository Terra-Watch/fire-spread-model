"""Training sample for the image models, drawn by a fixed rule BEFORE any image model is trained (see SAMPLING_PROTOCOL.md).
Population: data/big_cases.parquet. Only cases of the 17 v4 training states with label year <= 2023 are sampled; validation (2024), test (2025), blind and outside cases are never thinned (keep = True).
Rule: a fire event is kept when hash(SEED|event) < p_state, p_state = min(1, CAP / (USABLE_SHARE * selected training cases of the state)); all cases of a kept event stay together.
u_frac (hash of a second key) gives nested subsets for the learning curve (keep and u_frac < f). Writes data/sample_train.parquet and data/sample_plan.json (rule, per-state p, counts, sha256 of the kept case list).
Run: .venv/bin/python scripts/42_sample_train.py"""
import hashlib
import json
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
SEED, CAP, USABLE_SHARE = 20261105, 200, 0.35      # 0.35 = share of built bank cases that pass the burned-patch test (692 of 1,992 measured on 2026-11-05)
norm = lambda s: "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)).lower()
u = lambda key: int(hashlib.sha1(key.encode()).hexdigest()[:8], 16) / 2 ** 32


def groups():
    import re
    src = (ROOT / "scripts" / "60_train.py").read_text()
    pick = lambda name: {x.strip().strip('"') for x in re.search(name + r" = \{(.*?)\}", src, re.S).group(1).replace("\n", " ").split(",") if x.strip()}
    return pick("V4_TRAIN"), pick("V4_BLIND")


def main():
    T, B = groups(); c = pd.read_parquet(ROOT / "data" / "big_cases.parquet")
    c["st"] = c.state.map(norm); c["year"] = pd.to_datetime(c.t1).dt.year
    c["group"] = np.where(c.st.isin(B), "blind", np.where(c.st.isin(T), "train17", "outside"))
    c["pool"] = np.where(c.group != "train17", c.group, np.where(c.year <= 2023, "train", np.where(c.year == 2024, "validation", "test")))
    n_state = c[c.pool == "train"].groupby("st").size(); p = (CAP / (USABLE_SHARE * n_state)).clip(upper=1.0)
    c["p"] = c.st.map(p).fillna(1.0); c["u"] = [u(f"{SEED}|{e}") for e in c.event]; c["u_frac"] = [u(f"frac|{e}") for e in c.event]
    c["keep"] = (c.pool != "train") | (c.u < c.p)
    out = c[["case_id", "event", "state", "st", "year", "pool", "keep", "u_frac"]].sort_values("case_id"); out.to_parquet(ROOT / "data" / "sample_train.parquet", index=False)
    kept = out[(out.pool == "train") & out.keep]; digest = hashlib.sha256("\n".join(kept.case_id).encode()).hexdigest()
    plan = {"seed": SEED, "cap_per_state": CAP, "usable_share": USABLE_SHARE, "population": int(len(c)), "pools": c.pool.value_counts().to_dict(),
            "train_selected_by_state": n_state.to_dict(), "p_by_state": p.round(3).to_dict(), "train_cases_kept": int(len(kept)), "expected_usable_train": round(float(len(kept) * USABLE_SHARE)),
            "kept_by_state": kept.groupby("st").size().to_dict(), "kept_by_year": kept.groupby("year").size().to_dict(), "kept_case_list_sha256": digest}
    json.dump(plan, open(ROOT / "data" / "sample_plan.json", "w"), indent=1)
    print(json.dumps({k: plan[k] for k in ("train_cases_kept", "expected_usable_train", "kept_by_state", "kept_by_year", "kept_case_list_sha256")}, indent=1))


if __name__ == "__main__":
    main()
