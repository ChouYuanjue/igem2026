import hashlib
import importlib.util
import json
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('bime_asset_resolver',Path(__file__).parents[1]/'resolve_bime_asset.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

def fixture_index(tmp_path):
    p=tmp_path/'results/current.json';p.parent.mkdir();p.write_text('{}')
    index={'claims':{'example':{'primary':'results/current.json','sha256':hashlib.sha256(p.read_bytes()).hexdigest()}},'superseded':{}}
    target=tmp_path/'reproducibility/bime_rank/canonical.json';target.parent.mkdir(parents=True);target.write_text(json.dumps(index))
    return p,target,index

def test_resolves_verified_asset(tmp_path):
    p,_,_=fixture_index(tmp_path)
    assert module.resolve('example',tmp_path)==p

def test_rejects_overwritten_result(tmp_path):
    p,_,_=fixture_index(tmp_path);p.write_text('{"changed":true}')
    with pytest.raises(ValueError,match='changed'):module.resolve('example',tmp_path)

def test_rejects_superseded_parent(tmp_path):
    _,target,index=fixture_index(tmp_path);index['superseded']['results']='replacement';target.write_text(json.dumps(index))
    with pytest.raises(ValueError,match='Superseded'):module.resolve('example',tmp_path)

def test_unknown_claim_has_no_latest_fallback(tmp_path):
    fixture_index(tmp_path)
    with pytest.raises(KeyError):module.resolve('unknown',tmp_path)
