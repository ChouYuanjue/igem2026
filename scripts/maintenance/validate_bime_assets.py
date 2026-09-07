"""Cheap on-disk integrity checks; no inference or benchmark execution."""
import csv
import hashlib
import json
from pathlib import Path
import zipfile
import yaml
from resolve_bime_asset import ROOT, resolve

def main():
    index=json.loads((ROOT/'reproducibility/bime_rank/canonical.json').read_text())
    for key in index['claims']:resolve(key)
    route=yaml.safe_load(resolve('production').read_text())
    candidate=yaml.safe_load((ROOT/'configs/production_routes/bime_rank_candidate_v1.yaml').read_text())
    assert route['routes']==candidate['routes']
    assert route['deployments']==candidate['deployments']
    assert route['policies']==candidate['policies']
    checked=set()
    def walk(value):
        if isinstance(value,dict):
            for pathkey,hashkey,filename in [('ranker_bundle','ranker_sha256','ranker.json'),('protein_asset','protein_manifest_sha256','manifest.json'),('reaction_asset','reaction_manifest_sha256','manifest.json')]:
                if pathkey in value and hashkey in value:
                    p=ROOT/value[pathkey]/filename
                    assert hashlib.sha256(p.read_bytes()).hexdigest()==value[hashkey],str(p)
                    checked.add(str(p.relative_to(ROOT)))
            for v in value.values():walk(v)
        elif isinstance(value,list):
            for v in value:walk(v)
    walk(route)
    for x in route['deployments'].values():assert (ROOT/x).is_dir(),x
    fairness=json.loads((ROOT/'results/clipzyme_native_extension_v1/CURRENT_FAIR_COMPARISON.json').read_text())['provenance']
    for direction in ['r2e','e2r']:
        assert hashlib.sha256((ROOT/fairness[direction+'_summary']).read_bytes()).hexdigest()==fairness[direction+'_sha256']
    enzyme405=json.loads(resolve('enzyme405').read_text())
    assert hashlib.sha256(Path(enzyme405['pairs']).read_bytes()).hexdigest()==enzyme405['pairs_sha256']
    p=resolve('wetlab_success_first').parent
    summary=json.loads(resolve('wetlab_success_first').read_text())
    def rows(name):
        with (p/name).open() as f:return list(csv.DictReader(f))
    assert len(rows('MANUAL_SUCCESS_FIRST_EXPERIMENT_PLAN_20_ROWS.csv'))==summary['sheet_rows']==20
    assert len(rows('MANUAL_SUCCESS_FIRST_RECOMMENDATIONS_17.csv'))==summary['unique_reactions']==17
    for name,n in [('PRIMARY_BUILD_SET_16_UNIQUE.fasta',16),('HEDGE_BUILD_SET_27_UNIQUE.fasta',27)]:
        assert sum(x.startswith('>') for x in (p/name).read_text().splitlines())==n
    package=p/'BiME_Rank_WetLab_Prediction_Package_20260906'
    package_summary=json.loads((package/'summary.json').read_text())
    for key in ['primary_fasta','hedge_fasta']:assert (package/package_summary[key]).is_file()
    with zipfile.ZipFile(p/(package.name+'.zip')) as z:
        assert z.testzip() is None
        for f in package.iterdir():
            if f.is_file():assert z.read(package.name+'/'+f.name)==f.read_bytes()
    result={'canonical_claims':len(index['claims']),'route_assets_hash_verified':len(checked),'route_candidate_semantic_equivalence':True,'wetlab_rows':20,'wetlab_reactions':17,'fasta_constructs':[16,27],'zip_integrity':True,'inference_or_experiments_run':False}
    (ROOT/'reproducibility/bime_rank/validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
if __name__=='__main__':main()
