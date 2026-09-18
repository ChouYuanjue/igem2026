from pathlib import Path
import numpy as np,pandas as pd
from projects.active.terpene_screening.runtime.ranking_metrics import evaluate_full_candidate_scores
def test_full_metric_known_ranks():
 ids=['R3','R1','R2','R4']; scores=np.array([.2,.9,.8,.1]); m=evaluate_full_candidate_scores(scores,ids,{'R2','R4'}); assert m['best_positive_rank']==2; assert m['hit_at_1']==0 and m['hit_at_3']==1; assert abs(m['reciprocal_rank']-.5)<1e-12
def test_evaluator_is_no_training_and_full_support():
 s=(Path(__file__).resolve().parents[3]/'projects/active/fibre/runtime/reaction_support.py').read_text(); assert 'assert len(common)==11081' in s; assert 'model_training_performed' in s; assert 'evaluate_full_candidate_scores' in s; assert 'dev_pairs.csv' in s
