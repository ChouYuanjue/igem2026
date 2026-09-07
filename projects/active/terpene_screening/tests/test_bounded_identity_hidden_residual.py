import torch
from projects.active.terpene_screening.rank_open_world import BoundedIdentityHiddenResidualReactionDualTower
from projects.active.terpene_screening.train_cleanroom_identity_aux_residual import configure_r2e_identity_residual_trainables
from projects.active.terpene_screening.train_dual_tower_cold import ModelConfig,TerpeneDualTower

def _config(): return ModelConfig(protein_input_dim=4,reaction_input_dim=5,hidden_dim=7,embedding_dim=3,dropout=0.0)

def test_bounded_residual_is_exact_identity_at_zero_init_and_only_aux_is_trainable_after_configure():
    torch.manual_seed(1); cfg=_config(); base=TerpeneDualTower(cfg).eval(); model=BoundedIdentityHiddenResidualReactionDualTower(cfg,2,.1).eval(); model.load_base_state(base.state_dict())
    x=torch.randn(11,7); assert torch.equal(model.encode_reactions(x),base.encode_reactions(x[:,:5]))
    configure_r2e_identity_residual_trainables(model)
    assert [n for n,p in model.named_parameters() if p.requires_grad]==['aux_to_hidden.weight']

def test_bounded_residual_caps_hidden_displacement_norm():
    torch.manual_seed(2); cfg=_config(); model=BoundedIdentityHiddenResidualReactionDualTower(cfg,2,.075).eval()
    with torch.no_grad(): model.aux_to_hidden.weight.fill_(100.0)
    x=torch.randn(13,7); base=x[:,:5]; aux=x[:,5:]; net=model.base_reaction_tower.network
    with torch.no_grad():
        h=net[1](net[0](base)); r=model.aux_to_hidden(aux); scale=torch.clamp((.075*h.norm(dim=1,keepdim=True))/r.norm(dim=1,keepdim=True).clamp_min(1e-12),max=1.0); bounded=r*scale
    assert float((bounded.norm(dim=1)/h.norm(dim=1).clamp_min(1e-12)).max()) <= .0750001

def test_invalid_cap_rejected():
    cfg=_config()
    for cap in [0.0,-.1,1.1]:
        try: BoundedIdentityHiddenResidualReactionDualTower(cfg,2,cap)
        except ValueError: pass
        else: raise AssertionError(cap)
