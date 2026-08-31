from projects.active.terpene_screening.prepare_broad_rhea_difficulty_slices import (
    degree_bucket, protein_identity_bucket, reaction_similarity_bucket,
    resolve_reaction_feature_dir,
)


def test_difficulty_bucket_boundaries_are_explicit():
    assert protein_identity_bucket(0.19) == "lt20"
    assert protein_identity_bucket(0.20) == "20_40"
    assert protein_identity_bucket(0.40) == "40_60"
    assert protein_identity_bucket(0.80) == "ge80"
    assert reaction_similarity_bucket(0.29) == "lt0p3"
    assert reaction_similarity_bucket(0.30) == "0p3_0p5"
    assert reaction_similarity_bucket(0.70) == "0p7_0p9"
    assert reaction_similarity_bucket(0.90) == "ge0p9"
    assert degree_bucket(0) == "degree0"
    assert degree_bucket(1) == "degree1"
    assert degree_bucket(5) == "degree2_5"
    assert degree_bucket(20) == "degree6_20"
    assert degree_bucket(21) == "degree21plus"


def test_reaction_similarity_counts_intersection_not_boolean_any():
    import numpy as np
    from projects.active.terpene_screening.prepare_broad_rhea_difficulty_slices import nearest_train_reaction_similarity
    features = np.asarray([
        [1, 1, 1, 0],
        [1, 1, 1, 0],
        [1, 0, 0, 1],
    ], dtype=np.float32)
    out = nearest_train_reaction_similarity(
        ["Q"], ["SAME", "OTHER"], features=features,
        reaction_ids=["Q", "SAME", "OTHER"], drfp_dim=4,
    )
    row = out.iloc[0]
    assert row.nearest_train_reaction_id == "SAME"
    assert row.max_train_drfp_tanimoto == 1.0
    assert row.reaction_similarity_bucket == "ge0p9"


def test_reaction_feature_dir_can_bind_extended_schema(tmp_path):
    universe = tmp_path / "universe"
    explicit = tmp_path / "rdkitplus"
    assert resolve_reaction_feature_dir(universe, explicit) == explicit.resolve()
    assert resolve_reaction_feature_dir(universe, None) == (
        universe.resolve() / "reaction_features" / "drfp_categorical_v1"
    )
