from __future__ import annotations

import ast
import json
import subprocess
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reproducibility/bime_rank/source_roles.json"
ACTIVE_PREFIX = "projects/active/terpene_screening/"
REPRO_PREFIX = "reproducibility/bime_rank/"
SOURCE_PREFIXES = (ACTIVE_PREFIX, REPRO_PREFIX)

RUNTIME_SEEDS = [
    ACTIVE_PREFIX + "runtime/cli.py",
    ACTIVE_PREFIX + "runtime/reaction_to_enzyme.py",
    ACTIVE_PREFIX + "runtime/enzyme_to_reaction.py",
    ACTIVE_PREFIX + "runtime/backend_routing.py",
]

REPRODUCTION_SEEDS = [
    REPRO_PREFIX + "scripts/build_general_candidate_universe.py",
    REPRO_PREFIX + "scripts/rebuild_reactzyme_transfer_assets.py",
    REPRO_PREFIX + "scripts/rebuild_horizyn_distillation_preprocessing.py",
    REPRO_PREFIX + "scripts/pretrain_horizyn_reaction_feature_distillation.py",
    REPRO_PREFIX + "scripts/train_marts_adapted_production.py",
    REPRO_PREFIX + "scripts/train_marts_horizyn_exact_residual_production.py",
    REPRO_PREFIX + "scripts/prepare_production_dual_kernel_assets.py",
    REPRO_PREFIX + "scripts/train_cleanroom_directional_identity_aux_residual.py",
    REPRO_PREFIX + "scripts/run_e2r_anchored_lambdamart_v3_production_experts.py",
    REPRO_PREFIX + "scripts/build_enzgfm_protein_features.py",
    REPRO_PREFIX + "scripts/combine_protein_feature_blocks.py",
    REPRO_PREFIX + "scripts/build_general_reaction_features.py",
    REPRO_PREFIX + "scripts/build_rdkitplus_augmented_reaction_features.py",
    ACTIVE_PREFIX + "runtime/reaction_features.py",
    ACTIVE_PREFIX + "runtime/protein_embeddings.py",
    REPRO_PREFIX + "scripts/merge_protein_feature_libraries.py",
    REPRO_PREFIX + "scripts/run_bime_r2e_seed_context_v1.py",
    REPRO_PREFIX + "scripts/run_bime_e2r_seed_context_v1.py",
]


def tracked() -> set[str]:
    out = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT)
    return {x.decode() for x in out.split(b"\0") if x}


def source_python(tracked_paths: set[str]) -> dict[str, Path]:
    return {
        rel: ROOT / rel
        for rel in tracked_paths
        if rel.endswith(".py") and rel.startswith(SOURCE_PREFIXES) and "/archive/" not in rel
    }


def module_name(rel: str) -> str:
    local = rel[:-3].replace("/", ".")
    if local.endswith(".__init__"):
        local = local[: -len(".__init__")]
    return local


def module_map(files: dict[str, Path]) -> dict[str, str]:
    return {module_name(rel): rel for rel in files if module_name(rel)}


def imported_source_files(rel: str, files: dict[str, Path], modules: dict[str, str]) -> set[str]:
    try:
        tree = ast.parse(files[rel].read_text(encoding="utf-8", errors="ignore"))
    except SyntaxError:
        return set()
    current = module_name(rel)
    package = current.rsplit(".", 1)[0] if "." in current else ""
    found: set[str] = set()

    def resolve(name: str) -> None:
        candidate = name
        while candidate:
            if candidate in modules:
                found.add(modules[candidate])
                return
            candidate = candidate.rsplit(".", 1)[0] if "." in candidate else ""

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                resolve(alias.name)
        elif isinstance(node, ast.ImportFrom):
            name = node.module or ""
            if node.level:
                parts = package.split(".") if package else []
                cut = max(0, len(parts) - node.level + 1)
                prefix = ".".join(parts[:cut])
                name = ".".join(x for x in (prefix, name) if x)
            resolve(name)
            for alias in node.names:
                if alias.name != "*":
                    resolve(".".join(x for x in (name, alias.name) if x))
    return found


def closure(seeds: list[str], files: dict[str, Path], modules: dict[str, str]) -> set[str]:
    missing = [x for x in seeds if x not in files]
    if missing:
        raise FileNotFoundError("source-role seed(s) missing or untracked: " + ", ".join(missing))
    seen: set[str] = set()
    queue = deque(seeds)
    while queue:
        rel = queue.popleft()
        if rel in seen:
            continue
        seen.add(rel)
        queue.extend(imported_source_files(rel, files, modules) - seen)
    return seen


def main() -> None:
    tracked_paths = tracked()
    files = source_python(tracked_paths)
    modules = module_map(files)
    runtime = closure(RUNTIME_SEEDS, files, modules)

    provenance = json.loads((ROOT / "reproducibility/bime_rank/canonical_source_provenance.json").read_text())
    provenance_seeds = {
        str(source["path"])
        for claim in provenance["claims"].values()
        for source in claim.get("sources", [])
        if source.get("retain_in_reproduction_source")
        and str(source.get("path", "")).endswith(".py")
        and str(source.get("path", "")) in files
    }
    reproduction_seed_union = sorted(set(REPRODUCTION_SEEDS) | provenance_seeds)
    reproduction = closure(reproduction_seed_union, files, modules)

    release = json.loads((ROOT / "reproducibility/research_release_manifest.json").read_text())
    declared_tests = [x for x in release["validation"]["release_regression_suite"] if x.endswith(".py")]
    missing_release = [x for x in declared_tests if x not in tracked_paths]
    if missing_release:
        raise FileNotFoundError("release tests missing: " + ", ".join(sorted(missing_release)))
    release_tests = set(declared_tests)

    repro_tests = {
        rel for rel in files
        if rel.startswith(REPRO_PREFIX + "tests/") and Path(rel).name.startswith("test_")
    }
    active_formal_tests = {
        rel for rel in files
        if rel.startswith(ACTIVE_PREFIX + "tests/") and Path(rel).name.startswith("test_")
    }
    extended_tests = (repro_tests | active_formal_tests) - release_tests
    support_source = {rel for rel in files if rel.startswith(REPRO_PREFIX + "support/")}

    payload = {
        "schema_version": 2,
        "release_branch": "master",
        "method_identity": "BiME-Rank (reproduction namespace only)",
        "policy": {
            "active_tree": "Current product/research implementation only; archive and historical baseline records are excluded by namespace.",
            "current_runtime": "AST import closure from responsibility-named runtime entrypoints in the active package.",
            "canonical_reproduction": "Historical BiME-Rank reproduction seeds plus retained canonical provenance sources and their source import closure.",
            "release_regression": "Portable regression boundary declared by reproducibility/research_release_manifest.json.",
            "extended_reproduction_tests": "Tracked reproduction tests plus current formal scientific tests outside the portable release suite.",
            "historical_research_source": "Tracked support source under reproducibility/bime_rank/support; never a current runtime authority.",
            "historical_lineage_tests": "Lineage-only material belongs in archive and is deliberately absent from this source-role graph.",
            "deletion_inference": "Role classification never authorizes deletion; archive is retained lineage.",
        },
        "runtime_seeds": RUNTIME_SEEDS,
        "reproduction_seeds": reproduction_seed_union,
        "canonical_provenance_reproduction_seeds": sorted(provenance_seeds),
        "current_runtime": sorted(runtime),
        "canonical_reproduction": sorted(reproduction - runtime),
        "release_regression": sorted(release_tests),
        "extended_reproduction_tests": sorted(extended_tests),
        "historical_research_source": sorted(support_source),
        "historical_lineage_tests": [],
        "counts": {
            "tracked_active_and_reproduction_python": len(files),
            "current_runtime": len(runtime),
            "canonical_reproduction_exclusive": len(reproduction - runtime),
            "release_regression_tests": len(release_tests),
            "extended_reproduction_tests": len(extended_tests),
            "reproduction_support_source": len(support_source),
        },
    }
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload["counts"], indent=2))


if __name__ == "__main__":
    main()
