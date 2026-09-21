from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from scripts.starase_navigator.errors import AppError
from scripts.starase_navigator.open_world_inputs import detect_direct_open_world_inputs
from scripts.starase_navigator.serve import NavigatorRuntime


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVENTS = ROOT / "results/starase_navigator_runtime/run_events.jsonl"


def read_events(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def user_text(row: dict[str, Any]) -> str:
    payload = row.get("input") if isinstance(row.get("input"), dict) else {}
    return str(payload.get("text") or payload.get("user_text") or "").strip()


def _is_successful_scientific_step(row: dict[str, Any]) -> bool:
    if str(row.get("event_type") or "") != "run_step":
        return False
    if str(row.get("status") or "") != "success":
        return False
    output = row.get("output") if isinstance(row.get("output"), dict) else {}
    direction = str(output.get("direction") or "")
    return bool(user_text(row)) and direction in {
        "reaction_to_enzyme",
        "enzyme_to_reaction",
        "route_design",
        "pathway_compatibility",
    }


def semantics_preserving_variants(text: str) -> list[dict[str, str]]:
    """Generate generic surface-form changes without entity/task-specific rules."""
    value = str(text or "").strip()
    if not value:
        return []
    variants: list[tuple[str, str]] = [("original", value)]
    parsed = detect_direct_open_world_inputs(value)
    has_literal_payload = bool(parsed.protein_sequences or parsed.reaction)

    if not has_literal_payload:
        collapsed = re.sub(r"\s+", " ", value).strip()
        variants.append(("whitespace_normalized", collapsed))

        zh = bool(re.search(r"[\u3400-\u9fff]", value))
        if zh and not value.startswith(("请", "麻烦")):
            variants.append(("polite_prefix", "请按以下要求处理：" + value))
        elif not zh and not value.casefold().startswith(("please ", "could you ", "can you ")):
            variants.append(("polite_prefix", "Please " + value))

        stripped = value.rstrip("。！？!?.,，；; ")
        if stripped:
            variants.append(("terminal_punctuation", stripped + ("。" if zh else ".")))

    # Database identifiers are case-insensitive at the resolver boundary. Change only
    # the identifier token, never surrounding free text or literal biochemical payload.
    def normalize_db_id(match: re.Match[str]) -> str:
        token = match.group(0)
        if ":" in token:
            prefix, suffix = token.split(":", 1)
            return prefix.lower() + ":" + suffix
        return token.lower()

    id_variant = re.sub(
        r"(?i)\b(?:RHEA:\d+|CHEBI:\d+|MED:\d+|PMC\d+|PF\d{5})\b",
        normalize_db_id,
        value,
    )
    if id_variant != value:
        variants.append(("database_id_case", id_variant))

    result: list[dict[str, str]] = []
    seen: set[str] = set()
    for transform, candidate in variants:
        candidate = candidate.strip()
        if candidate and candidate not in seen:
            seen.add(candidate)
            result.append({"transform": transform, "text": candidate})
    return result


def semantic_fingerprint(result: dict[str, Any]) -> dict[str, Any]:
    """Extract scientific state while intentionally ignoring tool order and prose."""
    protein = (
        result.get("protein_resolution")
        if isinstance(result.get("protein_resolution"), dict)
        else {}
    )
    reaction = (
        result.get("reaction_resolution")
        if isinstance(result.get("reaction_resolution"), dict)
        else {}
    )

    positive_enzymes: list[str] = []
    for group in result.get("positive_enzyme_resolutions") or []:
        if not isinstance(group, dict):
            continue
        for row in group.get("candidates") or []:
            if isinstance(row, dict):
                value = str(row.get("id") or row.get("accession") or "").strip()
                if value:
                    positive_enzymes.append(value)

    positive_reactions: list[str] = []
    for group in result.get("positive_reaction_resolutions") or []:
        if not isinstance(group, dict):
            continue
        for row in group.get("candidates") or []:
            if isinstance(row, dict):
                value = str(row.get("rhea_id") or row.get("id") or "").strip()
                if value:
                    positive_reactions.append(value)

    constraints = (
        result.get("reaction_constraints")
        if isinstance(result.get("reaction_constraints"), dict)
        else {}
    )

    def normalized_groups(key: str) -> list[list[str]]:
        groups: list[list[str]] = []
        for group in constraints.get(key) or []:
            if not isinstance(group, dict):
                continue
            ids = sorted(
                {
                    str(row.get("chebi_id") or "").strip().upper()
                    for row in group.get("alternatives") or []
                    if isinstance(row, dict) and str(row.get("chebi_id") or "").strip()
                }
            )
            if ids:
                groups.append(ids)
        return sorted(groups)

    return {
        "direction": str(result.get("direction") or ""),
        "protein": {
            "id": str(protein.get("recommended_id") or ""),
            "mode": str(protein.get("mode") or ""),
        },
        "reaction": {
            "id": str(reaction.get("recommended_id") or ""),
            "mode": str(reaction.get("mode") or ""),
        },
        "positive_enzyme_ids": sorted(set(positive_enzymes)),
        "positive_reaction_ids": sorted(set(positive_reactions)),
        "retrieval_plan": dict(result.get("retrieval_plan") or {}),
        "target_conditions": dict(result.get("target_conditions") or {}),
        "required_substrate_groups": normalized_groups("required_substrate_groups"),
        "required_product_groups": normalized_groups("required_product_groups"),
    }


def historical_cases(
    events: list[dict[str, Any]],
    *,
    context_turns: int,
    limit: int,
) -> list[dict[str, Any]]:
    by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in events:
        if str(row.get("event_type") or "") == "run_step" and user_text(row):
            by_session[str(row.get("session_id") or "")].append(row)

    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    for session_id, rows in by_session.items():
        for index, row in enumerate(rows):
            if not _is_successful_scientific_step(row):
                continue
            target = user_text(row)
            variants = semantics_preserving_variants(target)
            if len(variants) < 2:
                continue
            prefix: list[str] = []
            for earlier in rows[max(0, index - context_turns):index]:
                text = user_text(earlier)
                if text and (not prefix or prefix[-1] != text):
                    prefix.append(text)
            digest = hashlib.sha256(
                json.dumps([prefix, target], ensure_ascii=False).encode("utf-8")
            ).hexdigest()[:16]
            if digest in seen:
                continue
            seen.add(digest)
            cases.append(
                {
                    "case_id": f"metamorphic-{digest}",
                    "source_session_id": session_id,
                    "context": prefix,
                    "variants": variants,
                }
            )
            if limit > 0 and len(cases) >= limit:
                return cases
    return cases


def run_variant(
    runtime: NavigatorRuntime,
    *,
    case_id: str,
    context: list[str],
    variant: dict[str, str],
    ui_language: str,
) -> dict[str, Any]:
    transform = str(variant["transform"])
    session_id = f"eval-{case_id}-{transform}"
    for prefix_turn in context:
        runtime.agent_resolve(
            prefix_turn, ui_language=ui_language, session_id=session_id
        )
    started = time.perf_counter()
    try:
        result = runtime.agent_resolve(
            str(variant["text"]),
            ui_language=ui_language,
            session_id=session_id,
        )
    except AppError as exc:
        return {
            "transform": transform,
            "text": str(variant["text"]),
            "status": "error",
            "error_code": str(exc.code or ""),
            "error_detail": str(exc.detail or "")[:1200],
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }
    return {
        "transform": transform,
        "text": str(variant["text"]),
        "status": "ok",
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        "fingerprint": semantic_fingerprint(result),
        "turn_count": int(
            ((result.get("agent_execution") or {}).get("turn_count") or 0)
            if isinstance(result.get("agent_execution"), dict)
            else 0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Metamorphic Starase agent evaluation derived from real successful run "
            "history. Surface paraphrases must preserve scientific execution state."
        )
    )
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--context-turns", type=int, default=3)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--ui-language", default="zh")
    parser.add_argument("--list-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    cases = historical_cases(
        read_events(args.events),
        context_turns=max(0, int(args.context_turns)),
        limit=max(0, int(args.limit)),
    )
    if args.list_only:
        print(json.dumps(cases, ensure_ascii=False, indent=2))
        return

    runtime = NavigatorRuntime()
    rows: list[dict[str, Any]] = []
    for case in cases:
        variants = [
            run_variant(
                runtime,
                case_id=str(case["case_id"]),
                context=list(case.get("context") or []),
                variant=dict(variant),
                ui_language=str(args.ui_language or "zh"),
            )
            for variant in case.get("variants") or []
        ]
        baseline = next(
            (
                row.get("fingerprint")
                for row in variants
                if row.get("transform") == "original" and row.get("status") == "ok"
            ),
            None,
        )
        for row in variants:
            row["fingerprint_matches_original"] = (
                baseline is not None
                and row.get("status") == "ok"
                and row.get("fingerprint") == baseline
            )
        rows.append({**case, "results": variants})

    variant_rows = [variant for case in rows for variant in case["results"]]
    non_original = [
        row for row in variant_rows if row.get("transform") != "original"
    ]
    report = {
        "schema": "starase-agent-metamorphic-v1",
        "case_count": len(rows),
        "variant_count": len(variant_rows),
        "non_original_variant_count": len(non_original),
        "matching_variant_count": sum(
            bool(row.get("fingerprint_matches_original")) for row in non_original
        ),
        "error_variant_count": sum(
            row.get("status") != "ok" for row in non_original
        ),
        "cases": rows,
        "llm_provenance": runtime.deepseek.provenance(),
    }
    encoded = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
