from projects.active.terpene_screening.rank_open_world import IdentityHiddenResidualReactionDualTower
from projects.active.terpene_screening.train_cleanroom_identity_aux_residual import configure_r2e_identity_residual_trainables
from projects.active.terpene_screening.train_dual_tower_cold import ModelConfig, TerpeneDualTower
import torch


def test_clean_residual_trains_only_auxiliary_projection_and_starts_as_identity() -> None:
    config=ModelConfig(protein_input_dim=5,reaction_input_dim=7,hidden_dim=6,embedding_dim=4,dropout=0.0)
    base=TerpeneDualTower(config).eval()
    model=IdentityHiddenResidualReactionDualTower(config,aux_input_dim=3).eval(); model.load_base_state(base.state_dict())
    trainable=configure_r2e_identity_residual_trainables(model)
    assert trainable == [model.aux_to_hidden.weight]
    assert [n for n,p in model.named_parameters() if p.requires_grad] == ["aux_to_hidden.weight"]
    values=torch.randn(11,7); aux=torch.randn(11,3)
    with torch.no_grad():
        expected=base.encode_reactions(values); actual=model.encode_reactions(torch.cat([values,aux],dim=1))
    assert torch.allclose(expected,actual,atol=1e-7,rtol=1e-7)
