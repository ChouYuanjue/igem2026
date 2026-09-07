from __future__ import annotations

import ast
import json
import subprocess
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROJECT_PREFIX = "projects/active/terpene_screening/"
OUT = ROOT / "reproducibility/bime_rank/source_roles.json"

RUNTIME_SEEDS = [
    PROJECT_PREFIX + "rank_open_world.py",
    PROJECT_PREFIX + "bime_rank_r2e_runtime.py",
    PROJECT_PREFIX + "bime_rank_e2r_runtime.py",
    PROJECT_PREFIX + "hierarchical_expert_routing.py",
]

# Reviewed, high-confidence source entrypoints needed to rebuild released databases,
# reproduce canonical evaluation families, or regenerate current conditional/context evidence.
# Dependencies imported by these files are added automatically below.
REPRODUCTION_SEEDS = [
    PROJECT_PREFIX + "build_general_candidate_universe.py",
    PROJECT_PREFIX + "build_enzgfm_protein_features.py",
    PROJECT_PREFIX + "build_general_reaction_features.py",
    PROJECT_PREFIX + "build_rdkitplus_augmented_reaction_features.py",
    PROJECT_PREFIX + "build_reaction_center_augmented_features.py",
    PROJECT_PREFIX + "extract_esmc_embeddings.py",
    PROJECT_PREFIX + "merge_protein_feature_libraries.py",
    PROJECT_PREFIX + "evaluate_bime_cost_aware_shortlist_retention_v1.py",
    # Canonical negative/confirmatory evidence is part of the release story too.
    PROJECT_PREFIX + "evaluate_bime_r2e_homology_context_retention_v1.py",
    PROJECT_PREFIX + "run_bime_r2e_reciprocal_consistency_v1.py",
    PROJECT_PREFIX + "run_bime_tps_cage_topk_expert_v1.py",
    PROJECT_PREFIX + "evaluate_locked_marts_dual_kernel_confirmatory.py",
    PROJECT_PREFIX + "evaluate_bime_r2e_seed_context_retention_v1.py",
    PROJECT_PREFIX + "evaluate_bime_e2r_seed_context_retention_v1.py",
    PROJECT_PREFIX + "evaluate_bime_multiseed_scaling_v1.py",
    PROJECT_PREFIX + "run_bime_r2e_seed_context_v1.py",
    PROJECT_PREFIX + "run_bime_e2r_seed_context_v1.py",
    PROJECT_PREFIX + "run_bime_r2e_clipzyme_expert_v1.py",
    PROJECT_PREFIX + "evaluate_e2r_clipzyme_v4_common_support.py",
    PROJECT_PREFIX + "evaluate_clipzyme_catalyst_common_support_v1.py",
    PROJECT_PREFIX + "evaluate_enzyme405_bime_augmented_v1.py",
    PROJECT_PREFIX + "audit_enzyme405_evidence_chain.py",
    PROJECT_PREFIX + "bootstrap_enzyme405_neural.py",
    PROJECT_PREFIX + "evaluate_orphan335_fixed_pool.py",
    PROJECT_PREFIX + "run_enzymecage_orphan335_author_retrieval.py",
    PROJECT_PREFIX + "evaluate_pure_cage_full_support_v1.py",
    PROJECT_PREFIX + "evaluate_marts_dual_kernel_against_production.py",
    PROJECT_PREFIX + "evaluate_locked_tps_multisource_route.py",
    PROJECT_PREFIX + "balance_wetlab_reactions_across_plates.py",
    PROJECT_PREFIX + "build_combined_wetlab_campaign.py",
    PROJECT_PREFIX + "build_wetlab_discovery_panels.py",
    PROJECT_PREFIX + "build_wetlab_plate_manifest.py",
    PROJECT_PREFIX + "manage_wetlab_feedback.py",
    PROJECT_PREFIX + "randomize_wetlab_candidate_positions.py",
]


def tracked() -> set[str]:
    out = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT)
    return {x.decode() for x in out.split(b"\0") if x}


def project_python(tracked_paths: set[str]) -> dict[str, Path]:
    return {
        rel: ROOT / rel
        for rel in tracked_paths
        if rel.startswith(PROJECT_PREFIX) and rel.endswith(".py")
    }


def module_map(files: dict[str, Path]) -> dict[str, str]:
    out: dict[str, str] = {}
    for rel in files:
        local = rel[len(PROJECT_PREFIX) : -3].replace("/", ".")
        if local.endswith(".__init__"):
            local = local[: -len(".__init__")]
        if local:
            out[local] = rel
            out["projects.active.terpene_screening." + local] = rel
    return out


def imported_project_files(rel: str, files: dict[str, Path], modules: dict[str, str]) -> set[str]:
    try:
        tree = ast.parse(files[rel].read_text(encoding="utf-8", errors="ignore"))
    except SyntaxError:
        return set()
    local_module = rel[len(PROJECT_PREFIX) : -3].replace("/", ".")
    package = local_module.rsplit(".", 1)[0] if "." in local_module else ""
    found: set[str] = set()

    def resolve(name: str) -> None:
        for candidate in (name, "projects.active.terpene_screening." + name):
            if candidate in modules:
                found.add(modules[candidate])

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                resolve(alias.name)
        elif isinstance(node, ast.ImportFrom):
            name = node.module or ""
            if node.level:
                parts = package.split(".") if package else []
                up = max(0, node.level - 1)
                prefix = ".".join(parts[: len(parts) - up])
                name = ".".join(x for x in (prefix, name) if x)
            resolve(name)
            for alias in node.names:
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
        for dep in imported_project_files(rel, files, modules):
            if dep not in seen:
                queue.append(dep)
    return seen


def main() -> None:
    tracked_paths = tracked()
    files = project_python(tracked_paths)
    modules = module_map(files)
    runtime = closure(RUNTIME_SEEDS, files, modules)
    provenance = json.loads((ROOT / "reproducibility/bime_rank/canonical_source_provenance.json").read_text())
    provenance_seeds = {
        source["path"]
        for claim in provenance["claims"].values()
        for source in claim.get("sources", [])
        if source.get("retain_in_reproduction_source")
        and str(source.get("path", "")).startswith(PROJECT_PREFIX)
        and str(source.get("path", "")).endswith(".py")
    }
    reproduction_seed_union = sorted(set(REPRODUCTION_SEEDS) | provenance_seeds)
    reproduction = closure(reproduction_seed_union, files, modules)

    release = json.loads((ROOT / "reproducibility/research_release_manifest.json").read_text())
    release_tests = {
        x
        for x in release["validation"]["release_regression_suite"]
        if x.startswith(PROJECT_PREFIX)
    }
    missing_tests = sorted(x for x in release_tests if x not in tracked_paths)
    if missing_tests:
        raise FileNotFoundError("release tests missing: " + ", ".join(missing_tests))

    current = runtime | reproduction | release_tests
    all_project = set(files)
    historical = all_project - current
    non_release_tests = {x for x in historical if "/tests/" in x}
    # Keep non-portable tests only when they directly exercise current runtime or
    # canonical/rebuild source. Tests of retired research branches belong outside
    # the public release Git surface and are recorded separately in the demotion audit.
    extended_reproduction_tests = {
        test
        for test in non_release_tests
        if imported_project_files(test, files, modules) & current
    }
    historical_lineage_tests = non_release_tests - extended_reproduction_tests
    historical_source = historical - non_release_tests

    payload = {
        "schema_version": 1,
        "release_branch": "master",
        "method_identity": "BiME-Rank",
        "policy": {
            "current_runtime": "AST import closure from reviewed runtime entrypoints; legacy filenames inside this closure are implementation provenance, not separate methods",
            "canonical_reproduction": "reviewed builders/evaluators plus their project-local import closure",
            "release_regression": "portable CI regression boundary",
            "extended_reproduction_tests": "non-portable tests retained because they directly import current runtime or canonical/rebuild source",
            "historical_research_source": "conservatively retained research lineage/tooling after index-only demotion of strictly isolated auxiliary scripts; not current runtime or numeric authority",
            "historical_lineage_tests": "tests that exercise only retired research branches; these are not part of the public release Git surface after demotion",
            "deletion_inference": "historical classification alone never authorizes deleting a server file",
        },
        "runtime_seeds": RUNTIME_SEEDS,
        "reproduction_seeds": reproduction_seed_union,
        "canonical_provenance_reproduction_seeds": sorted(provenance_seeds),
        "current_runtime": sorted(runtime),
        "canonical_reproduction": sorted(reproduction - runtime),
        "release_regression": sorted(release_tests),
        "extended_reproduction_tests": sorted(extended_reproduction_tests),
        "historical_research_source": sorted(historical_source),
        "historical_lineage_tests": sorted(historical_lineage_tests),
        "counts": {
            "tracked_project_python": len(all_project),
            "current_runtime": len(runtime),
            "canonical_reproduction_exclusive": len(reproduction - runtime),
            "release_regression_project_tests": len(release_tests),
            "extended_reproduction_tests": len(extended_reproduction_tests),
            "current_union": len(current),
            "historical_research_source": len(historical_source),
            "historical_lineage_tests": len(historical_lineage_tests),
        },
    }
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload["counts"], indent=2))


if __name__ == "__main__":
    main()
