from __future__ import annotations
import argparse
from pathlib import Path
import torch
from projects.active.terpene_screening import run_tps_active_site_xattn_v1 as base
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'results/tps_active_site_xattn_v1r1'

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('action',choices=['hpo'])
    ap.add_argument('--device',default='cuda')
    args=ap.parse_args()
    base.OUT=OUT
    base.hpo(torch.device(args.device))
if __name__=='__main__': main()
