from pathlib import Path
import shlex

ROOT = Path(workflow.basedir).parent.resolve()
OUT = Path(config["output_dir"])
DATA = config["dataset"]
PF = config["features"]["protein"]
RF = config["features"]["reaction"]
P_VIEWS = list(PF.get("views", []))
R_VIEWS = list(RF.get("views", []))
P_DISTANCE_VIEWS = list(PF.get("distance_views", []))
R_DISTANCE_VIEWS = list(RF.get("distance_views", []))
PRECOMPUTE = config.get("precompute", {})
RC = PRECOMPUTE.get("reaction_center", {})
RC_ENABLED = bool(RC.get("enabled", False))
GEOM = config.get("geometry", {})

VALIDATION = str(OUT / "validation.json")
PROTEIN_FEATURES = str(OUT / "_work" / "protein_features.csv")
REACTION_FEATURES = str(OUT / "_work" / "reaction_features.csv")
PROTEIN_AFFINITY = str(GEOM.get("protein_affinity", "") or "")
REACTION_AFFINITY = str(GEOM.get("reaction_affinity", "") or "")
BUNDLE = str(OUT / "reference_bundle")
MANIFEST = str(Path(BUNDLE) / "manifest.json")
RC_ROOT = OUT / "_work" / "reaction_center"
RC_WASSERSTEIN = str(RC_ROOT / "center_wasserstein_distance.npy")
RC_TOKENS = str(RC_ROOT / "center_token_jaccard_distance.npy")
RC_AVAILABLE = str(RC_ROOT / "center_available.npy")
RC_MANIFEST = str(RC_ROOT / "manifest.json")

if RC_ENABLED:
    R_DISTANCE_VIEWS.extend([
        {
            "name": "center_wasserstein",
            "distance": RC_WASSERSTEIN,
            "available": RC_AVAILABLE,
        },
        {
            "name": "center_tokens",
            "distance": RC_TOKENS,
            "available": RC_AVAILABLE,
        },
    ])


rule validate_dataset:
    input:
        proteins=DATA["proteins"],
        reactions=DATA["reactions"],
        pairs=DATA["positive_pairs"],
    output:
        VALIDATION,
    conda:
        "../envs/fibre-core.yaml"
    shell:
        """
        PYTHONPATH={ROOT} python3 -m projects.active.fibre.portable.validate           --proteins {input.proteins:q}           --reactions {input.reactions:q}           --pairs {input.pairs:q}           --output {output:q}
        """


if not PROTEIN_AFFINITY and PF["mode"] == "precomputed":
    rule protein_features:
        input:
            source=PF["path"],
            validation=VALIDATION,
        output:
            PROTEIN_FEATURES,
        conda:
            "../envs/fibre-core.yaml"
        shell:
            "mkdir -p $(dirname {output:q}) && cp {input.source:q} {output:q}"
elif not PROTEIN_AFFINITY and PF["mode"] == "esmc":
    rule protein_features:
        input:
            proteins=DATA["proteins"],
            validation=VALIDATION,
        output:
            PROTEIN_FEATURES,
        params:
            model=PF.get("model", "esmc_600m"),
        conda:
            "../envs/fibre-encoders.yaml"
        shell:
            """
            mkdir -p $(dirname {output:q})
            PYTHONPATH={ROOT} python3 -m projects.active.fibre.portable.features proteins               --proteins {input.proteins:q}               --output {output:q}               --model {params.model:q}               --device "$(printenv FIBRE_DEVICE || echo cpu)"               --cache-dir "$(printenv FIBRE_FEATURE_CACHE || echo results/fibre_feature_cache)"
            """
elif not PROTEIN_AFFINITY:
    raise ValueError("features.protein.mode must be precomputed or esmc")


if not REACTION_AFFINITY and RF["mode"] == "precomputed":
    rule reaction_features:
        input:
            source=RF["path"],
            validation=VALIDATION,
        output:
            REACTION_FEATURES,
        conda:
            "../envs/fibre-core.yaml"
        shell:
            "mkdir -p $(dirname {output:q}) && cp {input.source:q} {output:q}"
elif not REACTION_AFFINITY and RF["mode"] == "drfp":
    rule reaction_features:
        input:
            reactions=DATA["reactions"],
            validation=VALIDATION,
        output:
            REACTION_FEATURES,
        conda:
            "../envs/fibre-encoders.yaml"
        shell:
            """
            mkdir -p $(dirname {output:q})
            PYTHONPATH={ROOT} python3 -m projects.active.fibre.portable.features reactions               --reactions {input.reactions:q}               --output {output:q}
            """
elif not REACTION_AFFINITY:
    raise ValueError("features.reaction.mode must be precomputed or drfp")


def _factor_input(affinity, features):
    return affinity if affinity else features


if RC_ENABLED:
    rule build_reaction_center_views:
        input:
            reactions=DATA["reactions"],
            mapped=RC["mapped_reactions"],
            validation=VALIDATION,
        output:
            wasserstein=RC_WASSERSTEIN,
            tokens=RC_TOKENS,
            available=RC_AVAILABLE,
            manifest=RC_MANIFEST,
        conda:
            "../envs/fibre-chem.yaml"
        shell:
            """
            PYTHONPATH={ROOT} python3 -m projects.active.fibre.portable.reaction_center \
              --reactions {input.reactions:q} \
              --mapped-reactions {input.mapped:q} \
              --output-dir {RC_ROOT:q}
            """


rule build_reference_bundle:
    input:
        validation=VALIDATION,
        proteins=DATA["proteins"],
        reactions=DATA["reactions"],
        pairs=DATA["positive_pairs"],
        protein_geometry=_factor_input(PROTEIN_AFFINITY, PROTEIN_FEATURES),
        reaction_geometry=_factor_input(REACTION_AFFINITY, REACTION_FEATURES),
        protein_views=[x["path"] for x in P_VIEWS],
        reaction_views=[x["path"] for x in R_VIEWS],
        protein_distance_views=[
            p for x in P_DISTANCE_VIEWS for p in (x["distance"],x["available"])
        ],
        reaction_distance_views=[
            p for x in R_DISTANCE_VIEWS for p in (x["distance"],x["available"])
        ],
    output:
        MANIFEST,
    params:
        out=BUNDLE,
        graph_k=int(GEOM.get("graph_k", 0)),
        graph_backend=str(GEOM.get("backend", "auto")),
        graph_block_size=int(GEOM.get("block_size", 1024)),
        graph_dense_limit=int(GEOM.get("dense_limit", 5000)),
        ann_candidate_multiplier=int(GEOM.get("ann_candidate_multiplier", 8)),
        protein_view_args=" ".join(
            "--protein-view " + shlex.quote(str(x["name"]) + "=" + str(x["path"]))
            for x in P_VIEWS
        ),
        reaction_view_args=" ".join(
            "--reaction-view " + shlex.quote(str(x["name"]) + "=" + str(x["path"]))
            for x in R_VIEWS
        ),
        protein_distance_view_args=" ".join(
            "--protein-distance-view " + shlex.quote(
                str(x["name"]) + "=" + str(x["distance"]) + "," + str(x["available"])
            )
            for x in P_DISTANCE_VIEWS
        ),
        reaction_distance_view_args=" ".join(
            "--reaction-distance-view " + shlex.quote(
                str(x["name"]) + "=" + str(x["distance"]) + "," + str(x["available"])
            )
            for x in R_DISTANCE_VIEWS
        ),
        protein_arg=(
            f"--protein-affinity {PROTEIN_AFFINITY}"
            if PROTEIN_AFFINITY
            else f"--protein-features {PROTEIN_FEATURES}"
        ),
        reaction_arg=(
            f"--reaction-affinity {REACTION_AFFINITY}"
            if REACTION_AFFINITY
            else f"--reaction-features {REACTION_FEATURES}"
        ),
    conda:
        "../envs/fibre-core.yaml"
    shell:
        """
        PYTHONPATH={ROOT} python3 -m projects.active.fibre.portable.core_bundle           --proteins {input.proteins:q}           --reactions {input.reactions:q}           --pairs {input.pairs:q}           {params.protein_arg}           {params.reaction_arg}           {params.protein_view_args}           {params.reaction_view_args}           {params.protein_distance_view_args}           {params.reaction_distance_view_args}           --output-dir {params.out:q}           --graph-k {params.graph_k}           --graph-backend {params.graph_backend:q}           --graph-block-size {params.graph_block_size}           --graph-dense-limit {params.graph_dense_limit}           --ann-candidate-multiplier {params.ann_candidate_multiplier}
        """
