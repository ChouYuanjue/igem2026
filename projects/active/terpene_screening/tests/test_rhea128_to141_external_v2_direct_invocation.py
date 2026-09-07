import subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
def test_v2_evaluator_direct_script_invocation_loads_repo_package():
 p=subprocess.run([sys.executable,str(ROOT/'projects/active/terpene_screening/evaluate_rhea128_to141_external_v2.py'),'--help'],cwd=ROOT,capture_output=True,text=True)
 assert p.returncode==0, p.stderr
 assert '--benchmark-root' in p.stdout and '--candidate-root' in p.stdout
