from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.bridge.runtime.heterogeneous_fibre import FibreHCMRuntime


DEFAULT_CHECKPOINT = (
    ROOT
    / "results/fibre_heterogeneous_general_v1_seed20260723/fibre_hcm.pt"
)


def read_ids(path: Path | None) -> list[str]:
    if path is None:
        return []
    return [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Query a frozen FIBRE-HCM checkpoint on the registered general "
            "protein/reaction universe."
        )
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--asset-root", type=Path, default=ROOT)
    parser.add_argument("--direction", choices=("r2e", "e2r"), required=True)
    parser.add_argument("--query-id", required=True)
    parser.add_argument("--candidate-file", type=Path)
    parser.add_argument("--known-id", action="append", default=[])
    parser.add_argument("--known-file", type=Path)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device. Use cpu when CUDA is unavailable.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    candidate_ids = read_ids(args.candidate_file) or None
    known_ids = list(map(str, args.known_id)) + read_ids(args.known_file)
    runtime = FibreHCMRuntime(
        checkpoint=args.checkpoint.resolve(),
        asset_root=args.asset_root.resolve(),
        device=args.device,
    )

    if args.direction == "r2e":
        kwargs = {
            "candidate_ids": candidate_ids,
            "known_enzyme_ids": known_ids,
            "top_k": args.top_k,
        }
        if args.chunk_size is not None:
            kwargs["chunk_size"] = args.chunk_size
        frame = runtime.rank_reaction_to_enzyme(args.query_id, **kwargs)
        target_column = "protein_id"
    else:
        kwargs = {
            "candidate_ids": candidate_ids,
            "known_reaction_ids": known_ids,
            "top_k": args.top_k,
        }
        if args.chunk_size is not None:
            kwargs["chunk_size"] = args.chunk_size
        frame = runtime.rank_enzyme_to_reaction(args.query_id, **kwargs)
        target_column = "reaction_id"

    payload = {
        "method": "FIBRE-HCM",
        "direction": args.direction,
        "query_id": args.query_id,
        "checkpoint": str(args.checkpoint.resolve()),
        "candidate_scope": (
            str(args.candidate_file.resolve())
            if args.candidate_file is not None
            else "registered-general-universe"
        ),
        "known_context_count": len(known_ids),
        "top_k": len(frame),
        "results": [
            {
                "rank": int(row.rank),
                target_column: str(getattr(row, target_column)),
                "score": float(row.score),
            }
            for row in frame.itertuples(index=False)
        ],
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)


if __name__ == "__main__":
    main()
