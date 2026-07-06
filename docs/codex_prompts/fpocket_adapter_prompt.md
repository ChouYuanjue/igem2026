# Future fpocket Adapter Prompt

Implement `fpocket_to_enzymecage.py` under
`explorations/pocket/adapters/`.

The adapter should:

- run fpocket outside `external_repos/`
- parse fpocket pocket outputs
- create a pocket manifest compatible with `pocket_manifest.py`
- generate EnzymeCAGE-readable pocket inputs without modifying EnzymeCAGE
- connect to `run_compare_baselines.py`

Analysis goals:

- compare P2Rank and fpocket pocket center distance
- compare residue overlap
- compare EnzymeCAGE score shift
- identify rescued and harmed cases

Be conservative with external command signatures. If uncertain, record a TODO
and do not guess.
