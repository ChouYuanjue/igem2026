from __future__ import annotations

from pathlib import Path

import pytest

from projects.active.terpene_screening.core.engine import payload_to_argv
from projects.active.terpene_screening.core.input_audit import audit_protein_sequence
from projects.active.terpene_screening.core.routing import resolve_route
from projects.active.terpene_screening.rank_open_world import build_parser


def test_engine_payload_uses_the_production_parser():
    argv = payload_to_argv(
        "rank-reactions",
        {
            "enzyme_id": "TEST",
            "top_k": 10,
            "known_reaction_ids": ["R1", "R2"],
        },
    )
    args = build_parser().parse_args(argv)
    assert args.command == "rank-reactions"
    assert args.enzyme_id == "TEST"
    assert args.top_k == 10
    assert args.known_reaction_ids == ["R1", "R2"]


def test_engine_rejects_file_and_model_overrides_by_default():
    with pytest.raises(ValueError):
        payload_to_argv("rank-enzymes", {"reaction_id": "R1", "model_dir": "/tmp/model"})


def test_route_manifest_resolves_locked_top20_auxiliary():
    route = resolve_route(
        direction="enzyme_to_reaction",
        objective="top20",
        is_current=False,
    )
    assert route.route_id == "e2r-external-top20-dual-kernel-rrf-v1"
    assert route.auxiliary_deployment is not None
    assert route.auxiliary_deployment.name == "marts_dual_kernel_e2r_top20"


def test_strict_protein_input_rejects_invalid_sequence():
    with pytest.raises(ValueError):
        audit_protein_sequence("MABC*?", policy="strict")
