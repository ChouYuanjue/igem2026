from scripts.catalyst_finder.correspondence_snapshot import CorrespondenceSnapshot

def test_snapshot_separates_geometry_version_from_positive_registry_version():
    s=CorrespondenceSnapshot().as_dict()
    assert s['atlas_version']=='terpene-correspondence-deployment-atlas-v2'
    assert len(s['atlas_manifest_sha256'])==64
    assert len(s['omega_sha256'])==64
    assert s['omega_policy']=='production_all_verified_positives'
    assert s['atlas_manifest_sha256'] != s['omega_sha256']
