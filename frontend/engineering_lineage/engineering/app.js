const stages = [
  {
    id:"enzymecage", index:"01", title:"EnzymeCAGE", subtitle:"The root: improve structure-aware ranking inside a controlled enzyme candidate pool.",
    branches:[
      ["Reaction-similarity transfer","local","Similar chemistry may reuse enzyme-family knowledge.","Useful first candidate generator.","Kept as reaction-neighborhood evidence."],
      ["Gate matrix / candidate gate","turn","Shrink the expensive search space before structural ranking.","Worked in closed tasks, but excluded positives could never be recovered.","Forced us to treat candidate coverage as a first-class problem."],
      ["RF / HGB meta-ranking","reject","Fuse CAGE, reaction similarity and handcrafted evidence.","Useful on the original fixed pool, brittle outside it.","Dropped as the universal ranking core."],
      ["Rescue slots","turn","Reserve a few positions for strong evidence missed by the main order.","Local rescue worked.","Became an early ancestor of bounded correction."],
    ]
  },
  {
    id:"expansion", index:"02", title:"Candidate expansion", subtitle:"The first bottleneck was no longer ranking quality; it was whether the positive entered the pool at all.",
    branches:[
      ["Few-shot / seed homology expansion","keep","Known positives provide strong local biological context.","Reliable when seeds exist.","Retained as an explicit context mode."],
      ["MARTS external candidates","keep","Break out of the internal TPS candidate set.","Expanded both protein and reaction coverage.","Kept and used to motivate stricter external evaluation."],
      ["UniProt free merge","reject","Maximize discovery coverage by adding many new proteins.","Unlabelled candidates displaced useful Top-K positions.","Showed that unordered expansion is not enough."],
      ["Canonical prefix + tail quota","local","Expand the pool without destroying the trusted head.","Controlled rescue was stable.","Retained only as a bounded fallback."],
      ["Pfam single-domain gate","reject","Use TPS family-domain knowledge as a filter.","Shared domains admitted unrelated proteins.","Rejected as too coarse."],
      ["Exact Pfam architecture / reaction contract","local","Make family rules specific to full domain architecture and query chemistry.","Useful for rescue, audit and explanation.","Kept locally, not as a global hard gate."],
      ["Taxonomy scope","keep","Some users need explicit eukaryotic/prokaryotic boundaries.","Best handled as candidate-universe scope.","Retained without training another model."],
      ["Conformal / second-round candidate sets","local","Let candidate-set size reflect uncertainty and wet-lab budget.","Useful as coverage diagnostics.","Kept outside the primary ordering rule."],
    ]
  },
  {
    id:"broad", index:"03", title:"Broad Retrieval", subtitle:"Candidate generation becomes an ordered full-space retrieval problem.",
    branches:[
      ["ESM-C protein representation","keep","Encode arbitrary proteins beyond fixed family identifiers.","Provided broad open-protein features.","Retained in the Broad foundation."],
      ["DRFP + reaction categories","keep","Represent chemistry instead of database IDs.","Generalized to unseen reaction identities.","Retained on the reaction side."],
      ["Multi-positive dual tower","keep","Enzyme–reaction relations are many-to-many.","Produced the first durable open retrieval core.","Became Broad Retrieval."],
      ["PU / cluster masking","keep","Unlabelled and near-homologous pairs may be false negatives.","Reduced obvious false-negative pressure.","Retained in training."],
      ["Ordinary hard negatives","keep","Random negatives were too easy.","Improved useful discrimination.","Retained as training support."],
      ["Dedicated hard-negative curriculum","reject","Specialize aggressively for top-head confounders.","No stable replacement for the main model.","Dropped."],
      ["Top-K surrogate loss","reject","Optimize the actual wet-lab budget boundary directly.","Sensitive to K and false negatives.","Research-only; not in the main objective."],
    ]
  },
  {
    id:"tps-specialization", index:"04", title:"TPS biochemical specialization", subtitle:"We tested whether explicitly mechanistic TPS features could outperform broad representations.",
    branches:[
      ["Catalytic motif context","reject","DDxxD, NSE/DTE and DxDD motifs are mechanistically meaningful.","Local discrimination improved, global retrieval remained weak.","Rejected as a core ranker."],
      ["Motif RRF / residual","reject","Use motifs only as a second-stage correction.","Development gains did not survive freezing.","Frozen correction returned to zero."],
      ["P2Rank / pocket-local representation","reject","Catalytic pockets should be closer to substrate selectivity than whole sequence.","Some development gain; frozen confirmation failed.","Dropped."],
      ["Same-precursor / different-skeleton negatives","reject","Create realistic TPS confounders.","Semantically good, no stable final gain.","Dropped."],
      ["Mechanism auxiliary / skeleton metric","reject","Force representations to preserve precursor, topology and oxidation information.","Labels were too sparse/coarse.","Dropped."],
      ["Coarse skeleton classes","reject","Supervise product skeleton identity directly.","Occasional development wins, unstable freezing.","Dropped."],
      ["Morgan reaction clusters","reject","Use finer chemical clusters than coarse skeleton labels.","Benefit remained unstable.","Dropped."],
      ["Carbon-connectivity graph residual","reject","Model the carbon rearrangement that TPS enzymes actually create.","Frozen residual again converged toward zero.","Dropped."],
      ["TPS active-site cross-attention","reject","Learn reaction-to-active-site token interactions directly.","Hard-pool/residual/HPO route later exposed evaluation-alignment issues.","Did not become the broad main line."],
    ]
  },
  {
    id:"external", index:"05", title:"External pretrained transfer", subtitle:"Could a larger pretrained enzyme–reaction space replace or strengthen Broad?",
    branches:[
      ["Horizyn global MLNCE","reject","Transfer a larger pretrained enzyme–reaction space directly.","Did not stably beat Broad.","Dropped."],
      ["Frozen Horizyn reaction encoder","reject","Keep pretrained reaction geometry fixed and adapt the other side.","Insufficient ranking quality.","Dropped."],
      ["ESM-C → ProtT5 bridge","reject","Align protein spaces expected by different pretrained systems.","Representation alignment did not solve ranking.","Dropped."],
      ["DRFP + Horizyn concatenation","reject","Combine local chemistry and external reaction features directly.","Scale and interaction were unstable.","Dropped."],
      ["Reaction distillation residual","local","Preserve Broad and import only external-model increments.","Useful under restricted R2E budgets.","Kept locally; established the residual principle."],
    ]
  },
  {
    id:"graph-fusion", index:"06", title:"Graph, geometry & fusion", subtitle:"A parallel search for ways to add relational structure without surrendering robust ranking.",
    branches:[
      ["Geometry alignment","reject","Align reaction-space and protein-space internal geometry.","Promiscuity, convergent function and sparse labels broke one-to-one geometry.","Dropped as a production core."],
      ["Graph diffusion / multi-hop","reject","Propagate known function through the enzyme–reaction graph.","Hub amplification and poor double-cold entry.","Dropped."],
      ["Candidate-hub normalization","reject","Suppress proteins that score highly for nearly every query.","Multiple normalizers gave no stable gain.","Dropped."],
      ["Dual kernel","local","Require support from both protein and reaction neighborhoods.","Current-only freeze failed; restricted MARTS E2R Top-20 worked.","Kept only in a narrow budget."],
      ["Raw score addition","reject","Use the simplest possible ensemble.","Score scales were incompatible.","Dropped."],
      ["Percentile / tied-rank fusion","local","Normalize scales before fusion.","Useful only in selected settings.","Kept locally."],
      ["Reciprocal Rank Fusion","local","Fuse relative order instead of raw score.","Stable in selected E2R budgets.","Kept locally."],
      ["Fixed three-source fusion","turn","Exploit complementary unique hits across Pfam, kernels and experts.","Development improved; frozen performance fell.","Showed complementarity does not justify fixed global weights."],
      ["Three-seed ensemble","keep","Reduce optimization variance.","Improved robustness.","Retained."],
      ["Reliability / abstention","keep","Expose risk rather than treating raw scores as probabilities.","Useful interpretation layer.","Retained outside the ranking score."],
      ["R2E ↔ E2R cycle consistency","local","Bidirectional rankings should support each other.","No stable new hits after confirmation.","Kept as a diagnostic, not a ranker."],
    ]
  },
  {
    id:"generalization", index:"07", title:"Generalization, forgetting & novelty", subtitle:"Broad needed to expand without destroying capabilities that already worked.",
    branches:[
      ["Directional continuation","turn","Absorb broader enzyme–reaction data while preserving old capability.","Established the retention problem clearly.","Led to explicit anti-forgetting experiments."],
      ["Historical replay / embedding anchor","local","Preserve old-domain representation during continuation.","Useful retention control.","Kept as a technique, not architecture."],
      ["LwF","reject","Preserve old model outputs while learning new data.","No decisive production advantage.","Dropped as architecture."],
      ["Score distillation","reject","Preserve old ranking behavior directly.","No decisive production advantage.","Dropped."],
      ["Margin-MSE","reject","Preserve pairwise ranking margins.","Did not outperform simpler controls.","Dropped."],
      ["RecAdam","local","Bias early continuation toward source parameters.","Useful as a low-forgetting training route.","Local technique only."],
      ["Checkpoint blending / WiSE-FT","local","Move back toward the source model when new-domain learning causes forgetting.","Found Pareto points.","Did not solve conditional expertise."],
      ["Fisher / RegMean / TIES / AdaMerging","reject","Merge domain-specific capabilities in parameter space.","Less stable than ranking-level organization.","Dropped."],
      ["Post-hoc domain routing","turn","Different models may be useful in different query regions.","Strong evidence that expert value is conditional.","Direct precursor to query-level gating."],
      ["Low-similarity novelty expert","reject","Create a specialist for reaction-novel queries.","Replay/routing lacked stable general benefit.","Dropped."],
      ["Functional-prototype residual","reject","Correct Broad from learned functional prototypes.","Failed formal screening.","Dropped."],
    ]
  },
  {
    id:"reaction-center", index:"08", title:"Reaction center & EnzGFM", subtitle:"The successful local pattern emerged: strong base first, limited correction second.",
    branches:[
      ["Reaction-center V1","reject","Represent only atoms and bonds that change.","Failed the hard-slice gate.","Reworked as a residual."],
      ["Reaction-center residual","turn","Make mechanistic evidence incremental and safely absent.","More stable than a standalone mechanism ranker.","Moved toward bounded correction."],
      ["Bounded reaction-center V3","keep","Cap how far local mechanism evidence may move Broad.","Passed confirmation.","Direct ancestor of BRIDGE bounded experts."],
      ["Native EnzGFM","keep","Introduce a strong pretrained enzyme–reaction model.","Strong baseline and expert source.","Retained."],
      ["EnzGFM + RDKit / RDKit+","keep","Add explicit reaction chemistry to foundation-model evidence.","Entered the durable multi-expert set.","Retained."],
      ["EnzGFM + reaction center","local","Combine foundation and local mechanism signals.","Useful expert source.","Kept locally."],
      ["Top-2000 pair reranker / bounded residual","turn","Correct only the head instead of rescoring the universe.","Strengthened the limited-prefix principle.","Fed directly into anchored E2R."],
      ["Difficulty-aware / center gate","turn","Run a second stage only when a query needs it and evidence exists.","Specific gates varied; the idea survived.","Became query-level applicability."],
    ]
  },
  {
    id:"bime", index:"09", title:"BiME-Rank", subtitle:"Heterogeneous models became explicit experts with admission, fallback and cost-aware execution.",
    branches:[
      ["R2E expert candidate union","keep","Let independent retrievers contribute candidates before fusion.","Improved broad support.","Retained."],
      ["R2E LambdaRank","keep","Learn how score, rank, agreement and novelty should combine.","Confirmed and frozen.","Retained as the R2E predecessor."],
      ["R2E similarity router","keep","Different novelty regions prefer different routes.","Deterministic fallback worked.","Retained."],
      ["E2R unrestricted four-expert LambdaRank","turn","Copy R2E multi-expert success to E2R.","Damaged a strong EnzGFM head ranking.","Forced protection of the base order."],
      ["Anchored LambdaMART / baseline rescue","keep","Protect the strong base and rerank only a limited prefix.","Passed confirmation.","Direct BRIDGE ancestor."],
      ["CLIPZyme structural expert","keep","Add 3D structural evidence.","Passed admission with missing-aware fallback.","Retained."],
      ["Seed-context expert","keep","Use known positives when the task provides them.","Conditionally admitted.","Retained."],
      ["Homology-context expert","reject","Expose homology as an explicit common expert.","Failed external retention.","Rejected."],
      ["Reciprocal-consistency expert","reject","Promote bidirectional agreement into a formal expert.","Failed external retention.","Rejected."],
      ["Generic CAGE Top-20 expert","turn","Reintroduce the mature structural model as a universal expert.","OOF ranking metrics fell.","Showed that expert value is not globally valid."],
      ["Cost-aware hierarchy","keep","Expensive experts cannot score the full candidate universe.","Cheap experts search broadly; expensive experts run on shortlist.","Retained."],
    ]
  },
  {
    id:"fibre", index:"10", title:"FIBRE detour", subtitle:"The strongest alternative hypothesis: replace modular expert assembly with one unified relational geometry.", className:"fibre",
    branches:[
      ["Biological relation stratification","historical","Represent richer biology than a binary pair.","Launched the unified-model line.","Archived as FIBRE provenance."],
      ["Tensor-product field","historical","Model reaction × enzyme interaction explicitly.","Expressive, hard to justify as the universal ranking core.","Archived."],
      ["Interaction Atlas","historical","Interpret experts as local charts of one catalytic space.","Coherent theory; weak production necessity.","Archived."],
      ["Context-restricted domains + atlas gluing","reject","Make local charts agree on overlaps.","Consistency assumptions were too strong.","Gluing loss removed."],
      ["Catalytic kernel / conditional kinetics","historical","Tie abstract geometry to real catalytic interpretation.","Improved interpretation, not ranking dominance.","Archived."],
      ["Fully symmetric joint potential","reject","Unify both retrieval directions in one symmetric score.","Lost useful direction-specific information.","Rejected."],
      ["Geometric-mean two-sided gates","reject","Use symmetric conditional gates.","Unstable across directions.","Rejected."],
      ["Symmetric linear mixture","reject","Simplify conditional combination.","Still unstable.","Rejected."],
      ["Gibbs / log-partition aggregation","reject","Use a soft thermodynamic-style aggregation.","No stable gain.","Rejected."],
      ["KL barycenter","reject","Fuse conditional expert distributions geometrically.","No stable gain.","Rejected."],
      ["Molecular joint-potential compensation","reject","Correct unified geometry with molecular evidence.","Insufficient.","Rejected."],
      ["Expert-variance fallback","reject","Fall back according to expert uncertainty.","R2E degraded.","Rejected."],
      ["Normalized conditional mixture","reject","Make conditional mixtures comparable across experts.","R2E degraded strongly.","Rejected."],
      ["Dimension-scaled consistency","reject","Normalize cross-view consistency by dimensionality.","No advantage over removing consistency.","Rejected."],
      ["Linear conditional expectation","local","Keep the most stable conditional aggregation.","Best-behaved FIBRE mode, still not a stronger universal core.","Historical local result."],
      ["Log-Mean-Exp","reject","Use a smoother conditional aggregation.","Development acceptable, frozen R2E failed.","Rejected."],
      ["Second-order variance correction","reject","Correct the mean with uncertainty curvature.","Failed development.","Rejected."],
      ["Reaction-center posterior mode","reject","Condition the latent mode on mechanistic posterior evidence.","Structurally cleaner, frozen R2E failed.","Rejected."],
      ["Scientific-evidence layer","turn","Anchor structure, mechanism and context on a stable core.","Missing=zero and additive anchoring survived.","Carried into BRIDGE."],
      ["Heterogeneous conditional modes","turn","Place EnzGFM, reaction center, CLIPZyme and seed context in one graph.","Unified execution worked; unified ranking core did not dominate.","Expert modularity survived."],
      ["End-to-end query-adaptive mixture","reject","Learn expert weights directly per query.","Damaged R2E.","Rejected."],
      ["E2R-only adaptive gate","reject","Restrict adaptation to the safer direction.","Shared training still hurt R2E.","Rejected."],
      ["Frozen-core post-hoc gate","turn","Freeze the strong core and learn only a light query gate.","Preserved R2E but gains were insufficient.","Validated the safer gating pattern."],
      ["ERAM relational core + UniMol","reject","Replace hand-designed geometry with a broad relational learner.","Trained and evaluated; still failed to replace Broad.","Rejected as universal core."],
      ["Pluggable adapters / frozen context plugin","turn","Add evidence without retraining a universal core.","Plugin + fallback behavior survived.","Carried into BRIDGE."],
    ]
  },
  {
    id:"return", index:"11", title:"Return to Broad", subtitle:"The problem was reframed as ranking authority: who is allowed to move the base order, and when?",
    branches:[
      ["Rebind evidence to Broad Core","keep","Broad remained the most reliable global order.","Universal ordering authority returned to Broad.","Retained."],
      ["Expert-as-pair-evidence","keep","Experts should provide local evidence rather than a new global geometry.","Produced a modular correction interface.","Retained."],
      ["Missing = neutral","keep","Unavailable evidence is absence of evidence, not negative evidence.","Removed hidden penalties from sparse experts.","Hard BRIDGE contract."],
      ["Direction-specific score permission","keep","One expert may be useful in R2E but unsafe in E2R, or vice versa.","Ranking authority became directional.","Retained."],
      ["Expert-type hierarchy","keep","Organize foundation, structure, mechanism, context and family evidence.","Made expert roles auditable.","Retained."],
      ["Dynamic router V4","turn","Make expert value query-dependent rather than globally fixed.","Established query-conditioned weighting.","Refined into explicit applicability permission."],
      ["Dynamic router V6 / applicability","keep","Separate reranking experts, evidence-only experts and silent experts.","Produced the final permission model.","Retained."],
    ]
  },
  {
    id:"specialists", index:"12", title:"CAGE & TPS return as specialists", subtitle:"The earliest domain knowledge comes back, now with explicit boundaries.",
    branches:[
      ["Broad → generic CAGE reranking","turn","Check whether CAGE failed only because the old pool was narrow.","Broad found many positives CAGE could not natively score.","Proved that structural scoreability is itself conditional."],
      ["P450 CAGE specialist","keep","Test CAGE inside a coherent family.","Clear local gain.","Retained under family applicability."],
      ["Phosphatase CAGE specialist","keep","Repeat the family-specific hypothesis.","Clear local gain.","Retained under family applicability."],
      ["Terpene CAGE specialist","keep","Recover local structural value in the original chemistry domain.","Local gain.","Retained as a specialist."],
      ["TPS specialist correction","keep","Reuse the earliest TPS-specific knowledge without narrowing the broad model.","Sparse activation only inside the TPS manifold.","Retained as a gated specialist."],
    ]
  },
  {
    id:"bridge", index:"13", title:"BRIDGE", subtitle:"Broad Retrieval with Inference-Driven Gated Experts.", className:"bridge",
    branches:[
      ["Broad owns the default order","keep","Open-world coverage needs one stable universal ordering authority.","The base remains usable with zero optional experts.","Core invariant."],
      ["Query-specific applicability","keep","Expert value varies by query, domain and evidence state.","Applicability is decided per query.","Core invariant."],
      ["Direction-specific permissions","keep","R2E and E2R do not share identical expert behavior.","Permissions are directional.","Core invariant."],
      ["Bounded corrections","keep","A useful local expert should not destroy a strong global order.","Experts receive limited correction authority.","Core invariant."],
      ["Local specialists remain local","keep","Sparse specialists can be valuable without improving the global average much.","Family CAGE and TPS experts stay scoped.","Core invariant."],
    ]
  }
];

const atlas = document.getElementById("engineeringAtlas");
const filters = document.getElementById("statusFilters");
const search = document.getElementById("branchSearch");
const focusMain = document.getElementById("focusMain");
let activeStatus = "all";
let mainOnly = false;

function esc(value){
  return String(value).replace(/[&<>"']/g, ch => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[ch]));
}

function renderCard(item, side, idx){
  const [title,status,motivation,result,next] = item;
  return `<article class="branch-card" data-status="${status}" data-side="${side}" data-search="${esc([title,motivation,result,next].join(" ").toLowerCase())}" id="branch-${side}-${idx}-${Math.random().toString(36).slice(2,8)}">
    <div class="card-top"><h4>${esc(title)}</h4><span class="status ${status}">${status}</span></div>
    <dl><dt>Why</dt><dd>${esc(motivation)}</dd><dt>Result</dt><dd>${esc(result)}</dd><dt>Then</dt><dd>${esc(next)}</dd></dl>
  </article>`;
}

function render(){
  atlas.innerHTML = stages.map((stage, sidx) => {
    const left = stage.branches.filter((_,i)=>i%2===0);
    const right = stage.branches.filter((_,i)=>i%2===1);
    return `<section class="stage ${stage.className||""}" data-stage="${stage.id}">
      <svg class="stage-svg" aria-hidden="true"></svg>
      <div class="branch-column left">${left.map((x,i)=>renderCard(x,"left",`${sidx}-${i}`)).join("")}</div>
      <div class="stage-core"><span class="stage-index">${stage.index}</span><div class="stage-anchor"></div><h3>${esc(stage.title)}</h3><p>${esc(stage.subtitle)}</p></div>
      <div class="branch-column right">${right.map((x,i)=>renderCard(x,"right",`${sidx}-${i}`)).join("")}</div>
    </section>`;
  }).join("");
  requestAnimationFrame(()=>{applyFilters(); drawAll();});
}

function applyFilters(){
  const q=search.value.trim().toLowerCase();
  document.querySelectorAll(".branch-card").forEach(card=>{
    const statusOk=activeStatus==="all"||card.dataset.status===activeStatus;
    const searchOk=!q||card.dataset.search.includes(q);
    card.classList.toggle("hidden",!(statusOk&&searchOk));
  });
  document.querySelectorAll(".stage").forEach(stage=>stage.classList.toggle("main-only",mainOnly));
  requestAnimationFrame(drawAll);
}

function drawStage(stage){
  const svg=stage.querySelector(".stage-svg");
  if(!svg||window.innerWidth<=900||mainOnly){if(svg)svg.innerHTML="";return;}
  const sr=stage.getBoundingClientRect();
  const anchor=stage.querySelector(".stage-anchor").getBoundingClientRect();
  const ax=anchor.left+anchor.width/2-sr.left;
  const ay=anchor.top+anchor.height/2-sr.top;
  svg.setAttribute("viewBox",`0 0 ${sr.width} ${sr.height}`);
  svg.innerHTML="";
  stage.querySelectorAll(".branch-card:not(.hidden)").forEach(card=>{
    const r=card.getBoundingClientRect();
    const isLeft=card.dataset.side==="left";
    const bx=(isLeft?r.right:r.left)-sr.left;
    const by=r.top+r.height/2-sr.top;
    const dx=Math.max(55,Math.abs(bx-ax)*.46);
    const c1x=isLeft?ax-dx:ax+dx;
    const c2x=isLeft?bx+dx*.45:bx-dx*.45;
    const path=document.createElementNS("http://www.w3.org/2000/svg","path");
    path.setAttribute("d",`M ${ax} ${ay} C ${c1x} ${ay}, ${c2x} ${by}, ${bx} ${by}`);
    if(card.dataset.status==="keep"||card.dataset.status==="turn")path.setAttribute("class","main-branch");
    svg.appendChild(path);
  });
}

function drawAll(){document.querySelectorAll(".stage").forEach(drawStage);}

filters.addEventListener("click",e=>{
  const btn=e.target.closest("button[data-status]"); if(!btn)return;
  activeStatus=btn.dataset.status;
  filters.querySelectorAll("button").forEach(b=>b.classList.toggle("active",b===btn));
  applyFilters();
});
search.addEventListener("input",applyFilters);
focusMain.addEventListener("click",()=>{
  mainOnly=!mainOnly;
  focusMain.textContent=mainOnly?"Show every branch":"Show main line only";
  applyFilters();
  if(mainOnly)document.getElementById("atlas").scrollIntoView({behavior:"smooth"});
});
window.addEventListener("resize",()=>requestAnimationFrame(drawAll));
window.addEventListener("load",drawAll);
render();
