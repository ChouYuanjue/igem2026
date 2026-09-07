from pathlib import Path
import json
import pandas as pd
ROOT=Path('/home/s241850073/igem2026'); OUT=ROOT/'results/requested_r2e20_bime_v2_20260906'; OLD=ROOT/'results/requested_r2e20_nju_lab_final_v2_20260813'; LAB=ROOT/'local_candidate_libraries/nju_lab_gbk_20260812/candidates'
# Success-first manual decisions after model + mechanism + feasibility review.
D={
'aristolochene':('NJU_LAB_8781a8b8576e7154752b','NJU_LAB_4450a0d4cb538cc2c755','HIGH','KEEP','A-tier; ESM-C gated rank 1 and CLIP gated rank 1; no known prokaryotic seed, so keep a mechanistically distinct backup.'),
'avermitilol':('NJU_LAB_63c207030bdbe47624b0','NJU_LAB_e176731d34d47ea07378','VERY_HIGH','OVERRIDE_KNOWN_POSITIVE_HOMOLOG','A-tier 334 aa class-I cyclase; MMseqs 76.8% identity with 99.4%/99.4% coverage to experimentally annotated avermitilol synthase Q82RR7. Exact-reaction homology is prioritized over model consensus under time pressure.'),
'beta_copaene':('NJU_LAB_5fed55393939d1971372','NJU_LAB_b8f97383b96a76fcf13e','MEDIUM_HIGH','KEEP','A-tier; stage-1 gated rank 1 and EnzGFM gated rank 2. No exact prokaryotic positive seed; nearest MMseqs reference is not used as functional evidence.'),
'beta_farnesene':('NJU_LAB_d781988931763cb9b157','NJU_LAB_ea96b1ba764147cebf25','MEDIUM_HIGH','KEEP','A-tier; ESM-C gated rank 1, EnzGFM rank 4 with known-positive context. Backup has strong model support but is slightly short (249 aa), so stays backup.'),
'cubebol':('NJU_LAB_d999b822c96ecda07825','NJU_LAB_6a145e1c402caea6d5ca','HIGH','OVERRIDE_FEASIBILITY','A-tier 310 aa candidate has stronger gated Top-1 robustness than the automatic 673 aa B-tier winner while Top-3 robustness is essentially tied. Prefer lower expression/architecture risk.'),
'delta_cadinene':('NJU_LAB_007b20d0074a6fd8fac2','NJU_LAB_25c06aabbd887f0c13f1','HIGH','KEEP','A-tier; strong agreement across stage-1, exact ESM-C/EnzGFM, two-seed context and CLIP (gated CLIP rank 1). Second A-tier candidate is nearly tied and is the preferred hedge.'),
'dolabellatriene':('NJU_LAB_dacd03b9dd761a7f9dd7','NJU_LAB_8b6946b221114c7db0f4','MEDIUM_HIGH','OVERRIDE_TARGET_SPECIFIC_TEACHER','A-tier 478 aa class-I candidate; within motif-valid pool it is stage-1 rank 1 and has stronger BiME teacher/CLIP transfer than the automatic source-expert consensus choice. This also diversifies failure risk from the cembrene construct.'),
'epi_isozizaene':('NJU_LAB_7554584e9e7ae334d594','NJU_LAB_b3856d2fdb16d2d503d7','VERY_HIGH','OVERRIDE_KNOWN_POSITIVE_HOMOLOG','A-tier 361 aa candidate; 86.9% identity with full query/target coverage to experimentally validated epi-isozizaene synthase Q9K499. Exact-reaction near-homology is the strongest success-first evidence.'),
'epieremophilene':('NJU_LAB_55701a13df7c31df11c0','NJU_LAB_6a145e1c402caea6d5ca','MEDIUM','OVERRIDE_FEASIBILITY','No prokaryotic exact-reaction seed. Prefer A-tier 445 aa class-I candidate with ESM-C gated rank 1 and CLIP rank 3 over 673 aa B-tier automatic winner carrying length/hydrophobic flags.'),
'gleenol':('NJU_LAB_ff7bcddcbf47a5d1bd33','NJU_LAB_07430b09a2b3e2d4bfdd','MEDIUM_LOW','KEEP','A-tier class-I candidate and best gated consensus available, but no prokaryotic same-reaction seed and no CLIP signal. Treat as exploratory and keep a second A-tier construct if possible.'),
'isocembrene':('NJU_LAB_8b6946b221114c7db0f4','NJU_LAB_6a145e1c402caea6d5ca','MEDIUM_HIGH','OVERRIDE_SHARED_PRODUCT_PROFILE','Use the same A-tier 491 aa construct as S-cembrene A: row 20 is explicitly the side-product readout of row 19. It is stage-1 gated rank 1 for isocembrene and avoids the 673 aa B-tier automatic winner.'),
'pentalenene':('NJU_LAB_827388b49ddb70be2cd2','NJU_LAB_f077e3623ed365893163','VERY_HIGH','KEEP','A-tier; dominant gated robustness (Top-1 0.914, Top-3 0.992), exact EnzGFM rank 1 and known-positive seed rank 1. Strongest model-supported call in the panel.'),
's_cembrene_a':('NJU_LAB_8b6946b221114c7db0f4','NJU_LAB_6a145e1c402caea6d5ca','HIGH','OVERRIDE_FEASIBILITY_SHARED_PRODUCT_PROFILE','A-tier 491 aa construct is stage-1 gated rank 1, retains high known-positive EnzGFM similarity, and is strongly supported for both main cembrene A and isocembrene side-product readouts. Prefer over 673 aa B-tier winner.'),
'sativene':('NJU_LAB_06474210af70f816ce79','NJU_LAB_dacd03b9dd761a7f9dd7','HIGH','KEEP_WITH_MILD_FLAG','Very strong gated evidence (Top-1 0.848; EnzGFM rank 1; seed rank 1). B-tier is only a mild 19-aa hydrophobic-window flag (1.64), with normal 340 aa length and no severe/N-terminal membrane-like flag; retain as primary.'),
'sesquicineole':('NJU_LAB_ef531a782c5b7cb2bd6f','NJU_LAB_20b261291dcf5f6bde71','HIGH','KEEP','A-tier; EnzGFM gated rank 1, seed rank 1 and Top-3 robustness 0.962. Use a second motif-valid A-tier candidate as hedge if capacity allows.'),
'spiroviolene':('NJU_LAB_7f3a69bb571e2c538532','NJU_LAB_b5409ec9d1e9c4df41cc','HIGH','KEEP_BUT_IGNORE_MMSEQS_REF','A-tier; EnzGFM gated rank 1, seed rank 1 and Top-3 robustness 0.981. Its MMseqs reference P00183 is P450cam and is explicitly ignored as functional evidence; motif/seed/model evidence drives the choice. Backup chosen for strong CLIP/seed evidence without relying on that unrelated reference.'),
't_muurolol':('NJU_LAB_b3856d2fdb16d2d503d7','NJU_LAB_d0579a69c8a26ffc8201','VERY_HIGH','KEEP','A-tier; Top-3 robustness 1.0 with EnzGFM rank 1, seed rank 1 and CLIP rank 3. Backup is also A-tier and reaches Top-3 under all sampled weightings.'),
}
# Load automatic stage3 and source metadata.
auto=pd.read_csv(OUT/'final_recommendations_17.csv',dtype=str).fillna('').set_index('requested_group')
meta=pd.read_csv(LAB/'candidates_metadata.tsv.gz',sep='\t',dtype=str).fillna('').drop_duplicates('enzyme_id').set_index('enzyme_id')
old=pd.read_csv(OLD/'unique_reaction_rankings.csv',dtype=str).fillna(''); old['rank_num']=pd.to_numeric(old['rank'],errors='coerce'); old1=old.sort_values('rank_num').groupby('requested_group').first().candidate_id.to_dict()
base=pd.read_csv(OLD/'top10_predictions_20_rows_enriched.csv',dtype=str).fillna(''); t=base.groupby('requested_group',sort=False).first().reset_index().set_index('requested_group')
rows=[]
for g,(primary,backup,conf,decision,why) in D.items():
    ap=auto.loc[g]
    # pull model evidence from stage3 gated files if candidate is present
    gd=pd.read_csv(OUT/f'final_gated_rank_{g}.csv',dtype=str).fillna('').set_index('candidate_id')
    def evidence(cid):
        if cid not in gd.index:return {}
        r=gd.loc[cid]
        return {k:r.get(k,'') for k in ['gated_top1_robustness','gated_top3_robustness','gated_stage1_rank','gated_esmc_direct_rank','gated_enzgfm_direct_rank','gated_seed_rank','gated_clip_rank','enzgfm_seed_similarity','teacher_score','clip_transfer_score','feasibility_tier','severe_risk_flags','moderate_risk_flags']}
    ep=evidence(primary); eb=evidence(backup); m=meta.loc[primary]
    rows.append({'requested_group':g,'sheet_rows':str(t.loc[g]['number']) if 'number' in t.columns else '', 'substrate':t.loc[g]['substrate'],'product':t.loc[g]['product'],'manual_primary':primary,'manual_backup':backup,'qualitative_priority':conf,'manual_decision':decision,'automatic_stage3_primary':ap['primary'],'old_202608_primary':old1.get(g,''),'changed_vs_automatic':primary!=ap['primary'],'changed_vs_old':primary!=old1.get(g,''),'rationale':why,'primary_source_files':m.source_files,'primary_locus_tags':m.locus_tags,'primary_length':m.length,'primary_classI_motif':m.classI_motif,'primary_screening_sources':m.screening_sources,'primary_mmseqs_ref':m.mmseqs_ref_id,'primary_mmseqs_fident':m.mmseqs_fident,**{f'primary_{k}':v for k,v in ep.items()},**{f'backup_{k}':v for k,v in eb.items()}})
R=pd.DataFrame(rows)
# Preserve original sheet ordering by minimum listed row number from original reaction summary.
rs=pd.read_csv(OLD/'reaction_summary.csv',dtype=str).fillna('').set_index('requested_group')
R['first_sheet_row']=R.requested_group.map(lambda g:min(int(x) for x in rs.loc[g,'sheet_rows'].split(';')))
R=R.sort_values('first_sheet_row').reset_index(drop=True)
R.to_csv(OUT/'MANUAL_SUCCESS_FIRST_RECOMMENDATIONS_17.csv',index=False)
# 20-row exact plan
mp=R.set_index('requested_group'); plan=[]
for _,r in base[['number','substrate','product','requested_group']].drop_duplicates('number').sort_values('number',key=lambda s:pd.to_numeric(s)).iterrows():
    x=mp.loc[r.requested_group];plan.append({'number':r.number,'substrate':r.substrate,'product':r['product'],'requested_group':r.requested_group,'primary_candidate_id':x.manual_primary,'backup_candidate_id':x.manual_backup,'priority':x.qualitative_priority,'source_files':x.primary_source_files,'locus_tags':x.primary_locus_tags,'decision':x.manual_decision,'rationale':x.rationale})
P=pd.DataFrame(plan);P.to_csv(OUT/'MANUAL_SUCCESS_FIRST_EXPERIMENT_PLAN_20_ROWS.csv',index=False)
# Build-set FASTA/manifest. Main cembrene + isocembrene intentionally share a construct.
seq=meta.sequence.to_dict(); primary_ids=list(dict.fromkeys(R.manual_primary)); hedge_ids=list(dict.fromkeys(list(R.manual_primary)+list(R.manual_backup)))
def fasta(ids,path):
    with open(path,'w') as h:
        for cid in ids:
            h.write(f'>{cid}\n{seq[cid]}\n')
fasta(primary_ids,OUT/'PRIMARY_BUILD_SET_16_UNIQUE.fasta');fasta(hedge_ids,OUT/f'HEDGE_BUILD_SET_{len(hedge_ids)}_UNIQUE.fasta')
pd.DataFrame([{'candidate_id':c,'roles':';'.join(R.loc[R.manual_primary.eq(c),'requested_group'])} for c in primary_ids]).to_csv(OUT/'PRIMARY_BUILD_SET_16_UNIQUE.csv',index=False)
# concise review md
lines=['# Success-first manual audit — 20 wet-lab rows / 17 unique reactions','',f'- 17 unique reactions map to **{len(primary_ids)} unique primary constructs** because S-cembrene A and isocembrene intentionally share one construct.',f'- Primary+one-backup strategy deduplicates to **{len(hedge_ids)} unique constructs**.','- Qualitative priority and robustness are not calibrated activity probabilities.','- Manual overrides prioritize exact-reaction homology, product-forming class-I TPS architecture, target-specific BiME/CLIP evidence, and expression feasibility over generic source-model consensus.','']
for _,r in R.iterrows():
    lines += [f"## {r['product']} ({r['requested_group']})",f"- Primary: `{r.manual_primary}` — {r.qualitative_priority}",f"- Backup: `{r.manual_backup}`",f"- Decision: {r.manual_decision}",f"- Why: {r.rationale}",f"- Source: `{r.primary_source_files}` / `{r.primary_locus_tags}`",'']
(OUT/'MANUAL_SUCCESS_FIRST_REVIEW.md').write_text('\n'.join(lines),encoding='utf-8')
summary={'status':'complete','unique_reactions':17,'sheet_rows':20,'unique_primary_constructs':len(primary_ids),'unique_primary_plus_backup_constructs':len(hedge_ids),'manual_overrides':int(R.changed_vs_automatic.astype(bool).sum()),'primary_fasta':'PRIMARY_BUILD_SET_16_UNIQUE.fasta','hedge_fasta':f'HEDGE_BUILD_SET_{len(hedge_ids)}_UNIQUE.fasta','warning':'qualitative priority and robustness are not activity probabilities'}
(OUT/'MANUAL_SUCCESS_FIRST_SUMMARY.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(summary,indent=2,ensure_ascii=False));print(R[['requested_group','product','manual_primary','manual_backup','qualitative_priority','manual_decision','changed_vs_automatic']].to_string(index=False))
