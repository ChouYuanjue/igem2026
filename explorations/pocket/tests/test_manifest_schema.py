from explorations.pocket.adapters.pocket_manifest import (
    PocketRecord,
    read_manifest,
    write_manifest,
)


def test_manifest_roundtrip(tmp_path):
    records = [
        PocketRecord(
            run_id="demo",
            enzyme_id="A",
            structure_path="structures/A.pdb",
            pocket_method="p2rank_topk",
            pocket_source="p2rank",
            pocket_rank=1,
            pocket_global_id="A__p2rank__rank1",
            pocket_score=0.9,
            pocket_center_x=1.0,
            pocket_center_y=2.0,
            pocket_center_z=3.0,
            pocket_residues="A:10,A:11",
            pocket_pdb_path="pockets/A_rank1.pdb",
            source_raw_dir="raw/A",
            pocket_pdb_mode="cropped_pocket",
        ),
        PocketRecord(
            run_id="demo",
            enzyme_id="B",
            structure_path="structures/B.pdb",
            pocket_method="p2rank_topk",
            pocket_source="p2rank",
            pocket_rank=2,
            pocket_global_id="B__p2rank__rank2",
            pocket_score=None,
            pocket_center_x=None,
            pocket_center_y=None,
            pocket_center_z=None,
            pocket_residues="",
            pocket_pdb_path="pockets/B_rank2.pdb",
            source_raw_dir="raw/B",
            pocket_pdb_mode="full_structure_placeholder",
        ),
    ]

    manifest_path = tmp_path / "pocket_manifest.csv"
    write_manifest(records, manifest_path)
    loaded = read_manifest(manifest_path)

    assert len(loaded) == 2
    assert loaded[0] == records[0]
    assert loaded[1].enzyme_id == "B"
    assert loaded[1].pocket_score is None
    assert isinstance(loaded[1].pocket_rank, int)


if __name__ == "__main__":
    pass
