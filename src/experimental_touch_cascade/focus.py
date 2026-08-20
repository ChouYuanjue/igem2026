from __future__ import annotations

import pandas as pd


def load_focus(profile) -> pd.DataFrame:
    f = profile.focus
    if not f.ranking_path:
        return pd.DataFrame(columns=["candidate_id","focus_group","model_rank","force_stage"])
    d = pd.read_csv(f.ranking_path, dtype={f.candidate_column: str, f.group_column: str})
    d[f.rank_column] = pd.to_numeric(d[f.rank_column], errors="raise").astype(int)
    rows = []
    for stage, k in [(2, f.stage2_top_k_per_group), (3, f.stage3_top_k_per_group)]:
        x = d[d[f.rank_column] <= k][[f.candidate_column, f.group_column, f.rank_column]].copy()
        x.columns = ["candidate_id","focus_group","model_rank"]
        x["force_stage"] = stage
        rows.append(x)
    out = pd.concat(rows, ignore_index=True).drop_duplicates(["candidate_id","focus_group","force_stage"])
    return out.sort_values(["force_stage","model_rank","focus_group","candidate_id"]).reset_index(drop=True)
