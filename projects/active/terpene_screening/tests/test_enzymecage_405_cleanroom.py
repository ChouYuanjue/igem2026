from pathlib import Path

import pandas as pd
import torch

from projects.active.terpene_screening.evaluate_enzymecage_405_cleanroom import (
    filter_author_gvp_reservoir,
    filter_author_valid_pocket_reservoir,
)


def test_author_gvp_only_is_explicitly_diagnostic(tmp_path: Path) -> None:
    gvp = tmp_path / "gvp.pt"
    torch.save({"P1": (torch.zeros((2, 3)),), "P3": (torch.zeros((2, 3)),)}, gvp)
    frame = pd.DataFrame({
        "UniprotID": ["P1", "P2", "P3", "P2"],
        "reaction_id": ["R1", "R1", "R2", "R2"],
        "label": [1, 0, 1, 0],
    })
    filtered, audit = filter_author_gvp_reservoir(frame, gvp)
    assert filtered.UniprotID.tolist() == ["P1", "P3"]
    assert audit["mode"] == "author_gvp_only"
    assert "Diagnostic only" in audit["interpretation"]


def test_author_valid_pocket_filter_mirrors_gvp_esm_intersection_and_node_check(tmp_path: Path) -> None:
    gvp = tmp_path / "gvp.pt"
    esm = tmp_path / "esm.pt"
    torch.save({
        "P1": (torch.zeros((2, 3)),),
        "P2": (torch.zeros((2, 3)),),
        "P3": (torch.zeros((2, 3)),),
    }, gvp)
    torch.save({
        "P1": torch.zeros((2, 1152)),
        "P3": torch.zeros((3, 1152)),  # present but rejected by author node-count check
    }, esm)
    frame = pd.DataFrame({
        "UniprotID": ["P1", "P2", "P3"],
        "reaction_id": ["R1", "R1", "R2"],
        "label": [1, 0, 1],
    })
    filtered, audit = filter_author_valid_pocket_reservoir(frame, gvp, esm)
    assert filtered.UniprotID.tolist() == ["P1"]
    assert audit["gvp_uids"] == 3
    assert audit["esm_node_uids"] == 2
    assert audit["gvp_esm_intersection_uids"] == 2
    assert audit["node_count_mismatch_uids"] == 1
    assert audit["author_inference_valid_uids"] == 1
    assert audit["queries_without_positive_after_filter"] == 0


def test_author_valid_pocket_filter_reports_lost_positive_query(tmp_path: Path) -> None:
    gvp = tmp_path / "gvp.pt"
    esm = tmp_path / "esm.pt"
    torch.save({"N1": (torch.zeros((2, 3)),)}, gvp)
    torch.save({"N1": torch.zeros((2, 1152))}, esm)
    frame = pd.DataFrame({"UniprotID": ["P1", "N1"], "reaction_id": ["R1", "R1"], "label": [1, 0]})
    _, audit = filter_author_valid_pocket_reservoir(frame, gvp, esm)
    assert audit["queries_without_positive_after_filter"] == 1
