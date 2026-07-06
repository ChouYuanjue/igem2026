# EnzymeCAGE Asset Tree Summary

- generated_at: 2026-05-15T05:02:40.105661+00:00
- enzymecage_root: `external_repos/EnzymeCAGE`

## Checkpoints

- seed_42 `epoch_19.pth`: True
- seed_42 `best_model.pth`: False
- total .pth files: 20
  - `external_repos/EnzymeCAGE/checkpoints/domain-specific-ft/p450/seed_40/epoch_9.pth`
  - `external_repos/EnzymeCAGE/checkpoints/domain-specific-ft/p450/seed_41/epoch_9.pth`
  - `external_repos/EnzymeCAGE/checkpoints/domain-specific-ft/p450/seed_42/epoch_9.pth`
  - `external_repos/EnzymeCAGE/checkpoints/domain-specific-ft/p450/seed_43/epoch_9.pth`
  - `external_repos/EnzymeCAGE/checkpoints/domain-specific-ft/p450/seed_44/epoch_9.pth`
  - `external_repos/EnzymeCAGE/checkpoints/domain-specific-ft/phosphatase/seed_40/epoch_9.pth`
  - `external_repos/EnzymeCAGE/checkpoints/domain-specific-ft/phosphatase/seed_41/epoch_9.pth`
  - `external_repos/EnzymeCAGE/checkpoints/domain-specific-ft/phosphatase/seed_42/epoch_9.pth`
  - `external_repos/EnzymeCAGE/checkpoints/domain-specific-ft/phosphatase/seed_43/epoch_9.pth`
  - `external_repos/EnzymeCAGE/checkpoints/domain-specific-ft/phosphatase/seed_44/epoch_9.pth`
  - `external_repos/EnzymeCAGE/checkpoints/domain-specific-ft/terpene/seed_40/epoch_9.pth`
  - `external_repos/EnzymeCAGE/checkpoints/domain-specific-ft/terpene/seed_41/epoch_9.pth`
  - `external_repos/EnzymeCAGE/checkpoints/domain-specific-ft/terpene/seed_42/epoch_9.pth`
  - `external_repos/EnzymeCAGE/checkpoints/domain-specific-ft/terpene/seed_43/epoch_9.pth`
  - `external_repos/EnzymeCAGE/checkpoints/domain-specific-ft/terpene/seed_44/epoch_9.pth`
  - `external_repos/EnzymeCAGE/checkpoints/pretrain/seed_40/epoch_19.pth`
  - `external_repos/EnzymeCAGE/checkpoints/pretrain/seed_41/epoch_19.pth`
  - `external_repos/EnzymeCAGE/checkpoints/pretrain/seed_42/epoch_19.pth`
  - `external_repos/EnzymeCAGE/checkpoints/pretrain/seed_43/epoch_19.pth`
  - `external_repos/EnzymeCAGE/checkpoints/pretrain/seed_44/epoch_19.pth`

## Dataset Paths

- `dataset/internal-test-set`: True
- `dataset/internal-test-set/Enzyme-405`: True
- `dataset/internal-test-set/Orphan-335`: True
- `dataset/RHEA/2023-07-12`: True
- `dataset/RHEA/2025-02-05`: True
- `dataset/RHEA/2025-02-05/pockets/pocket`: True
- `dataset/demo`: False

## Candidate Input CSVs

### `external_repos/EnzymeCAGE/dataset/RHEA/2023-07-12/rhea_rxn2uids.csv`
- size_bytes: 344857894
- n_rows: 307027
- columns: RHEA_ID, DIRECTION, MASTER_ID, UniprotID, SMILES, EC number, CANO_RXN_SMILES, sequence, reverse_template, n_seq

```json
[
  {
    "RHEA_ID": "10008",
    "DIRECTION": "UN",
    "MASTER_ID": "10008",
    "UniprotID": "O17433",
    "SMILES": "*N[C@@H](CS)C(*)=O.*N[C@@H](CS)C(*)=O.*OO>>*N[C@@H](CSSC[C@H](N*)C(*)=O)C(*)=O.*O.[H]O[H]",
    "EC number": "",
    "CANO_RXN_SMILES": "*N[C@@H](CS)C(*)=O.*N[C@@H](CS)C(*)=O.*OO>>*N[C@@H](CSSC[C@H](N*)C(*)=O)C(*)=O.*O.O",
    "sequence": "MTKGILLGDKFPDFRAETNEGFIPSFYDWIGKDSWAILFSHPRDFTPVCTTELARLVQLAPEFKKRNVKLIGLSCDSAESHRKWVDDIMAVCKMKCNDGDTCCSGNKLPFPIIADENRFLATELGMMDPDERDENGNALTARCVFIIGPEKTLKLSILYPATTGRNFDEILRVVDSLQLTAVKLVATPVDWKGGDDCVVLPTIDDTEAKKLFGEKINTIELPSGKHYLRMVAHPK",
    "reverse_template": "[C;D1;H0:1]-[OH;D1;+0:2].[C:4]-[S;H0;D2;+0:5]-[S;H0;D2;+0:7]-[C:6].[OH2;D0;+0:3]>>[C;D1;H0:1]-[O;H0;D2;+0:2]-[OH;D1;+0:3].[C:4]-[SH;D1;+0:5].[C:6]-[SH;D1;+0:7]",
    "n_seq": "235"
  },
  {
    "RHEA_ID": "10008",
    "DIRECTION": "UN",
    "MASTER_ID": "10008",
    "UniprotID": "O34564",
    "SMILES": "*N[C@@H](CS)C(*)=O.*N[C@@H](CS)C(*)=O.*OO>>*N[C@@H](CSSC[C@H](N*)C(*)=O)C(*)=O.*O.[H]O[H]",
    "EC number": "",
    "CANO_RXN_SMILES": "*N[C@@H](CS)C(*)=O.*N[C@@H](CS)C(*)=O.*OO>>*N[C@@H](CSSC[C@H](N*)C(*)=O)C(*)=O.*O.O",
    "sequence": "MAERMVGKQAPRFEMEAVLASKEFGKVSLEENMKNDKWTVLFFYPMDFTFVCPTEITAMSDRYDEFEDLDAEVIGVSTDTIHTHLAWINTDRKENGLGQLKYPLAADTNHEVSREYGVLIEEEGVALRGLFIINPEGELQYQTVFHNNIGRDVDETLRVLQALQTGGLCPANWKPGQKTL",
    "reverse_template": "[C;D1;H0:1]-[OH;D1;+0:2].[C:4]-[S;H0;D2;+0:5]-[S;H0;D2;+0:7]-[C:6].[OH2;D0;+0:3]>>[C;D1;H0:1]-[O;H0;D2;+0:2]-[OH;D1;+0:3].[C:4]-[SH;D1;+0:5].[C:6]-[SH;D1;+0:7]",
    "n_seq": "180"
  },
  {
    "RHEA_ID": "10008",
    "DIRECTION": "UN",
    "MASTER_ID": "10008",
    "UniprotID": "P0C0L2",
    "SMILES": "*N[C@@H](CS)C(*)=O.*N[C@@H](CS)C(*)=O.*OO>>*N[C@@H](CSSC[C@H](N*)C(*)=O)C(*)=O.*O.[H]O[H]",
    "EC number": "",
    "CANO_RXN_SMILES": "*N[C@@H](CS)C(*)=O.*N[C@@H](CS)C(*)=O.*OO>>*N[C@@H](CSSC[C@H](N*)C(*)=O)C(*)=O.*O.O",
    "sequence": "MTIHKKGQAHWEGDIKRGKGTVSTESGVLNQQPYGFNTRFEGEKGTNPEELIGAAHAACFSMALSLMLGEAGFTPTSIDTTADVSLDKVDAGFAITKIALKSEVAVPGIDASTFDGIIQKAKAGCPVSQVLKAEITLDYQLKS",
    "reverse_template": "[C;D1;H0:1]-[OH;D1;+0:2].[C:4]-[S;H0;D2;+0:5]-[S;H0;D2;+0:7]-[C:6].[OH2;D0;+0:3]>>[C;D1;H0:1]-[O;H0;D2;+0:2]-[OH;D1;+0:3].[C:4]-[SH;D1;+0:5].[C:6]-[SH;D1;+0:7]",
    "n_seq": "143"
  }
]
```

### `external_repos/EnzymeCAGE/dataset/RHEA/2025-02-05/all_enzymes.csv`
- size_bytes: 75357970
- n_rows: 200392
- columns: UniprotID, sequence

```json
[
  {
    "UniprotID": "O17433",
    "sequence": "MTKGILLGDKFPDFRAETNEGFIPSFYDWIGKDSWAILFSHPRDFTPVCTTELARLVQLAPEFKKRNVKLIGLSCDSAESHRKWVDDIMAVCKMKCNDGDTCCSGNKLPFPIIADENRFLATELGMMDPDERDENGNALTARCVFIIGPEKTLKLSILYPATTGRNFDEILRVVDSLQLTAVKLVATPVDWKGGDDCVVLPTIDDTEAKKLFGEKINTIELPSGKHYLRMVAHPK"
  },
  {
    "UniprotID": "O34564",
    "sequence": "MAERMVGKQAPRFEMEAVLASKEFGKVSLEENMKNDKWTVLFFYPMDFTFVCPTEITAMSDRYDEFEDLDAEVIGVSTDTIHTHLAWINTDRKENGLGQLKYPLAADTNHEVSREYGVLIEEEGVALRGLFIINPEGELQYQTVFHNNIGRDVDETLRVLQALQTGGLCPANWKPGQKTL"
  },
  {
    "UniprotID": "P0C0L2",
    "sequence": "MTIHKKGQAHWEGDIKRGKGTVSTESGVLNQQPYGFNTRFEGEKGTNPEELIGAAHAACFSMALSLMLGEAGFTPTSIDTTADVSLDKVDAGFAITKIALKSEVAVPGIDASTFDGIIQKAKAGCPVSQVLKAEITLDYQLKS"
  }
]
```

### `external_repos/EnzymeCAGE/dataset/RHEA/2025-02-05/rhea_rxn2uids.csv`
- size_bytes: 522334388
- n_rows: 300476
- columns: RHEA_ID, DIRECTION, MASTER_ID, UniprotID, SMILES, EC number, CANO_RXN_SMILES, sequence, reverse_template, n_seq, rxnmapper_template, localmapper_template

```json
[
  {
    "RHEA_ID": "10008",
    "DIRECTION": "UN",
    "MASTER_ID": "10008",
    "UniprotID": "O17433",
    "SMILES": "*OO.*N[C@@H](CS)C(*)=O.*N[C@@H](CS)C(*)=O>>*N[C@@H](CSSC[C@H](N*)C(*)=O)C(*)=O.*O.[H]O[H]",
    "EC number": "",
    "CANO_RXN_SMILES": "*N[C@@H](CS)C(*)=O.*N[C@@H](CS)C(*)=O.*OO>>*N[C@@H](CSSC[C@H](N*)C(*)=O)C(*)=O.*O.O",
    "sequence": "MTKGILLGDKFPDFRAETNEGFIPSFYDWIGKDSWAILFSHPRDFTPVCTTELARLVQLAPEFKKRNVKLIGLSCDSAESHRKWVDDIMAVCKMKCNDGDTCCSGNKLPFPIIADENRFLATELGMMDPDERDENGNALTARCVFIIGPEKTLKLSILYPATTGRNFDEILRVVDSLQLTAVKLVATPVDWKGGDDCVVLPTIDDTEAKKLFGEKINTIELPSGKHYLRMVAHPK",
    "reverse_template": "[C;D1;H0:1]-[OH;D1;+0:2].[C:4]-[S;H0;D2;+0:5]-[S;H0;D2;+0:7]-[C:6].[OH2;D0;+0:3]>>[C;D1;H0:1]-[O;H0;D2;+0:2]-[OH;D1;+0:3].[C:4]-[SH;D1;+0:5].[C:6]-[SH;D1;+0:7]",
    "n_seq": "235",
    "rxnmapper_template": "[C;D1;H3:1]-[OH;D1;+0:2].[C:4]-[S;H0;D2;+0:5]-[S;H0;D2;+0:7]-[C:6].[OH2;D0;+0:3]>>[C;D1;H3:1]-[O;H0;D2;+0:2]-[OH;D1;+0:3].[C:4]-[SH;D1;+0:5].[C:6]-[SH;D1;+0:7]",
    "localmapper_template": "[C;D1;H3:1]-[OH;D1;+0:2].[C:4]-[S;H0;D2;+0:5]-[S;H0;D2;+0:7]-[C:6].[OH2;D0;+0:3]>>[C;D1;H3:1]-[O;H0;D2;+0:2]-[OH;D1;+0:3].[C:4]-[SH;D1;+0:5].[C:6]-[SH;D1;+0:7]"
  },
  {
    "RHEA_ID": "10008",
    "DIRECTION": "UN",
    "MASTER_ID": "10008",
    "UniprotID": "O34564",
    "SMILES": "*OO.*N[C@@H](CS)C(*)=O.*N[C@@H](CS)C(*)=O>>*N[C@@H](CSSC[C@H](N*)C(*)=O)C(*)=O.*O.[H]O[H]",
    "EC number": "",
    "CANO_RXN_SMILES": "*N[C@@H](CS)C(*)=O.*N[C@@H](CS)C(*)=O.*OO>>*N[C@@H](CSSC[C@H](N*)C(*)=O)C(*)=O.*O.O",
    "sequence": "MAERMVGKQAPRFEMEAVLASKEFGKVSLEENMKNDKWTVLFFYPMDFTFVCPTEITAMSDRYDEFEDLDAEVIGVSTDTIHTHLAWINTDRKENGLGQLKYPLAADTNHEVSREYGVLIEEEGVALRGLFIINPEGELQYQTVFHNNIGRDVDETLRVLQALQTGGLCPANWKPGQKTL",
    "reverse_template": "[C;D1;H0:1]-[OH;D1;+0:2].[C:4]-[S;H0;D2;+0:5]-[S;H0;D2;+0:7]-[C:6].[OH2;D0;+0:3]>>[C;D1;H0:1]-[O;H0;D2;+0:2]-[OH;D1;+0:3].[C:4]-[SH;D1;+0:5].[C:6]-[SH;D1;+0:7]",
    "n_seq": "180",
    "rxnmapper_template": "[C;D1;H3:1]-[OH;D1;+0:2].[C:4]-[S;H0;D2;+0:5]-[S;H0;D2;+0:7]-[C:6].[OH2;D0;+0:3]>>[C;D1;H3:1]-[O;H0;D2;+0:2]-[OH;D1;+0:3].[C:4]-[SH;D1;+0:5].[C:6]-[SH;D1;+0:7]",
    "localmapper_template": "[C;D1;H3:1]-[OH;D1;+0:2].[C:4]-[S;H0;D2;+0:5]-[S;H0;D2;+0:7]-[C:6].[OH2;D0;+0:3]>>[C;D1;H3:1]-[O;H0;D2;+0:2]-[OH;D1;+0:3].[C:4]-[SH;D1;+0:5].[C:6]-[SH;D1;+0:7]"
  },
  {
    "RHEA_ID": "10008",
    "DIRECTION": "UN",
    "MASTER_ID": "10008",
    "UniprotID": "P0C0L2",
    "SMILES": "*OO.*N[C@@H](CS)C(*)=O.*N[C@@H](CS)C(*)=O>>*N[C@@H](CSSC[C@H](N*)C(*)=O)C(*)=O.*O.[H]O[H]",
    "EC number": "",
    "CANO_RXN_SMILES": "*N[C@@H](CS)C(*)=O.*N[C@@H](CS)C(*)=O.*OO>>*N[C@@H](CSSC[C@H](N*)C(*)=O)C(*)=O.*O.O",
    "sequence": "MTIHKKGQAHWEGDIKRGKGTVSTESGVLNQQPYGFNTRFEGEKGTNPEELIGAAHAACFSMALSLMLGEAGFTPTSIDTTADVSLDKVDAGFAITKIALKSEVAVPGIDASTFDGIIQKAKAGCPVSQVLKAEITLDYQLKS",
    "reverse_template": "[C;D1;H0:1]-[OH;D1;+0:2].[C:4]-[S;H0;D2;+0:5]-[S;H0;D2;+0:7]-[C:6].[OH2;D0;+0:3]>>[C;D1;H0:1]-[O;H0;D2;+0:2]-[OH;D1;+0:3].[C:4]-[SH;D1;+0:5].[C:6]-[SH;D1;+0:7]",
    "n_seq": "143",
    "rxnmapper_template": "[C;D1;H3:1]-[OH;D1;+0:2].[C:4]-[S;H0;D2;+0:5]-[S;H0;D2;+0:7]-[C:6].[OH2;D0;+0:3]>>[C;D1;H3:1]-[O;H0;D2;+0:2]-[OH;D1;+0:3].[C:4]-[SH;D1;+0:5].[C:6]-[SH;D1;+0:7]",
    "localmapper_template": "[C;D1;H3:1]-[OH;D1;+0:2].[C:4]-[S;H0;D2;+0:5]-[S;H0;D2;+0:7]-[C:6].[OH2;D0;+0:3]>>[C;D1;H3:1]-[O;H0;D2;+0:2]-[OH;D1;+0:3].[C:4]-[SH;D1;+0:5].[C:6]-[SH;D1;+0:7]"
  }
]
```

### `external_repos/EnzymeCAGE/dataset/internal-test-set/Enzyme-405/Enzyme-405.csv`
- size_bytes: 33861512
- n_rows: 15921
- columns: RHEA_ID, DIRECTION, MASTER_ID, UniprotID, SMILES, EC number, CANO_RXN_SMILES, sequence, reverse_template, n_seq, rxnmapper_template, localmapper_template, Label, similar_rxn, rxn_similarity, n_enzymes

```json
[
  {
    "RHEA_ID": "81171",
    "DIRECTION": "UN",
    "MASTER_ID": "81171",
    "UniprotID": "A6ZZV7",
    "SMILES": "CCCCCC[C@@H](O)CCCCCC/C=C/[C@H](O)[C@@H](O)[C@H](O)[C@H]([NH3+])C(=O)[O-].CC(=O)SCCNC(=O)CCNC(=O)[C@H](O)C(C)(C)COP(=O)([O-])OP(=O)([O-])OC[C@H]1O[C@@H](N2C=NC3=C2N=CN=C3N)[C@H](O)[C@@H]1OP(=O)([O-])[O-]>>CCCCCC[C@@H](O)CCCCCC/C=C/[C@H](OC(",
    "EC number": "",
    "CANO_RXN_SMILES": "CC(=O)SCCNC(=O)CCNC(=O)[C@H](O)C(C)(C)COP(=O)(O)OP(=O)(O)OC[C@H]1O[C@@H](n2cnc3c(N)ncnc32)[C@H](O)[C@@H]1OP(=O)(O)O.CCCCCC[C@@H](O)CCCCCC/C=C/[C@H](O)[C@@H](O)[C@H](O)[C@H](N)C(=O)O>>CC(C)(COP(=O)(O)OP(=O)(O)OC[C@H]1O[C@@H](n2cnc3c(N)ncnc32",
    "sequence": "MSTIKPSPSNNNLKVRSRPRRKSSIGKIDLGDTVPSLGTMFETKESKTAAKRRMQRLSEATKNDSDLVKKIWFSFREISYRHAWIAPLMILIAVYSAYFTSGNTTKTNVLHRFVAVSYQIGDTNAYGKGINDLCFVFYYMIFFTFLREFLMDVVIRPFAIRLHVTSKHRIKRIMEQMYAIFYTGVSGPFGIYCMYHSDLWFFNTKAMYRTYPDFTNPFLFKVFYLGQAAFWAQQACILVL",
    "reverse_template": "[C:1]-[SH;D1;+0:2].[C:6]=[C:7]/[C:8]-[O;H0;D2;+0:9]-[C;H0;D3;+0:3](-[C;D1;H3:4])=[O;D1;H0:5]>>[C:1]-[S;H0;D2;+0:2]-[C;H0;D3;+0:3](-[C;D1;H3:4])=[O;D1;H0:5].[C:6]=[C:7]/[C:8]-[OH;D1;+0:9]",
    "n_seq": "515",
    "rxnmapper_template": "[C:1]-[SH;D1;+0:2].[C:6]=[C:7]/[C:8]-[O;H0;D2;+0:9]-[C;H0;D3;+0:3](-[C;D1;H3:4])=[O;D1;H0:5]>>[C:1]-[S;H0;D2;+0:2]-[C;H0;D3;+0:3](-[C;D1;H3:4])=[O;D1;H0:5].[C:6]=[C:7]/[C:8]-[OH;D1;+0:9]",
    "localmapper_template": "[C:1]-[SH;D1;+0:2].[C:6]=[C:7]/[C:8]-[O;H0;D2;+0:9]-[C;H0;D3;+0:3](-[C;D1;H3:4])=[O;D1;H0:5]>>[C:1]-[S;H0;D2;+0:2]-[C;H0;D3;+0:3](-[C;D1;H3:4])=[O;D1;H0:5].[C:6]=[C:7]/[C:8]-[OH;D1;+0:9]",
    "Label": "0",
    "similar_rxn": "*C(=O)SCCNC(=O)CCNC(=O)[C@H](O)C(C)(C)COP(=O)(O)OP(=O)(O)OC[C@H]1O[C@@H](n2cnc3c(N)ncnc32)[C@H](O)[C@@H]1OP(=O)(O)O.CCCCCCCCCCCCC/C=C/[C@@H](O)[C@@H](N)CO>>*C(=O)N[C@@H](CO)[C@H](O)/C=C/CCCCCCCCCCCCC.CC(C)(COP(=O)(O)OP(=O)(O)OC[C@H]1O[C@@H]",
    "rxn_similarity": "0.5778029205320978",
    "n_enzymes": "2.0"
  },
  {
    "RHEA_ID": "78495",
    "DIRECTION": "UN",
    "MASTER_ID": "78495",
    "UniprotID": "Q7VLN6",
    "SMILES": "NC1=NC(=O)N([C@@H]2O[C@H](COP(=O)([O-])OP(=O)([O-])CCCNO)[C@@H](O)[C@H]2O)C=C1.CC(=O)SCCNC(=O)CCNC(=O)[C@H](O)C(C)(C)COP(=O)([O-])OP(=O)([O-])OC[C@H]1O[C@@H](N2C=NC3=C2N=CN=C3N)[C@H](O)[C@@H]1OP(=O)([O-])[O-]>>CC(=O)N(O)CCCP(=O)([O-])OP(=O)",
    "EC number": "",
    "CANO_RXN_SMILES": "CC(=O)SCCNC(=O)CCNC(=O)[C@H](O)C(C)(C)COP(=O)(O)OP(=O)(O)OC[C@H]1O[C@@H](n2cnc3c(N)ncnc32)[C@H](O)[C@@H]1OP(=O)(O)O.Nc1ccn([C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)CCCNO)[C@@H](O)[C@H]2O)c(=O)n1>>CC(=O)N(O)CCCP(=O)(O)OP(=O)(O)OC[C@H]1O[C@@H](n2ccc(",
    "sequence": "MRQLVILSTMPPIQGGVLLNSQNFSKAKQFLGQEYPFALYDMRSENGICFNLEAFAIIVGTIQENGTLYLICPNWHSVEQQIDVDAIRWNGGVAIACPHFFQHFKRLINKFGFEVTSRPQQPFIKTAPSYPAKLIQFTDEQQNILQKLPLDPAEIHIITAARGRGKSTLAGKLAEQFAKTEQVILTAHRSSSIQKILQTASINIPFFAPDKLLNLIETKQISADHLLFIDEAACIPLPIL",
    "reverse_template": "[C:1]-[N;H0;D3;+0:2](-[O;D1;H1:3])-[C;H0;D3;+0:6](-[C;D1;H3:7])=[O;D1;H0:8].[C:4]-[SH;D1;+0:5]>>[C:1]-[NH;D2;+0:2]-[O;D1;H1:3].[C:4]-[S;H0;D2;+0:5]-[C;H0;D3;+0:6](-[C;D1;H3:7])=[O;D1;H0:8]",
    "n_seq": "286",
    "rxnmapper_template": "[C:1]-[N;H0;D3;+0:2](-[O;D1;H1:3])-[C;H0;D3;+0:6](-[C;D1;H3:7])=[O;D1;H0:8].[C:4]-[SH;D1;+0:5]>>[C:1]-[NH;D2;+0:2]-[O;D1;H1:3].[C:4]-[S;H0;D2;+0:5]-[C;H0;D3;+0:6](-[C;D1;H3:7])=[O;D1;H0:8]",
    "localmapper_template": "[C:1]-[N;H0;D3;+0:2](-[O;D1;H1:3])-[C;H0;D3;+0:6](-[C;D1;H3:7])=[O;D1;H0:8].[C:4]-[SH;D1;+0:5]>>[C:1]-[NH;D2;+0:2]-[O;D1;H1:3].[C:4]-[S;H0;D2;+0:5]-[C;H0;D3;+0:6](-[C;D1;H3:7])=[O;D1;H0:8]",
    "Label": "0",
    "similar_rxn": "*O[C@H]1[C@@H](O)[C@H](n2ccc(N)nc2=O)O[C@@H]1COP(*)(=O)O.CC(=O)SCCNC(=O)CCNC(=O)[C@H](O)C(C)(C)COP(=O)(O)OP(=O)(O)OC[C@H]1O[C@@H](n2cnc3c(N)ncnc32)[C@H](O)[C@@H]1OP(=O)(O)O.Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)(O)OP(=O)(O)OP(=O)(O)O)[C@@H](O)",
    "rxn_similarity": "0.6443085954347215",
    "n_enzymes": "2.0"
  },
  {
    "RHEA_ID": "80951",
    "DIRECTION": "UN",
    "MASTER_ID": "80951",
    "UniprotID": "Q0T128",
    "SMILES": "[1*][C@@H](O)CC(=O)NCC(=O)[O-].*N[C@@H](COP(=O)([O-])OCC(C)(C)[C@@H](O)C(=O)NCCC(=O)NCCSC(*)=O)C(*)=O>>[1*][C@H](CC(=O)NCC(=O)[O-])OC([2*])=O.*N[C@@H](COP(=O)([O-])OCC(C)(C)[C@@H](O)C(=O)NCCC(=O)NCCS)C(*)=O",
    "EC number": "",
    "CANO_RXN_SMILES": "*N[C@@H](COP(=O)(O)OCC(C)(C)[C@@H](O)C(=O)NCCC(=O)NCCSC(*)=O)C(*)=O.[1*][C@@H](O)CC(=O)NCC(=O)O>>*N[C@@H](COP(=O)(O)OCC(C)(C)[C@@H](O)C(=O)NCCC(=O)NCCS)C(*)=O.[1*][C@H](CC(=O)NCC(=O)O)OC([2*])=O",
    "sequence": "MLFSFFRNLCRVLYRVRVTGDTQALKGERVLITPNHVSFIDGILLGLFLPVRPVFAVYTSISQQWYMRWLKSFIDFVPLDPTQPMAIKHLVRLVEQGRPVVIFPEGRITTTGSLMKIYDGAGFVAAKSGATVIPVRIEGAELTHFSRLKGLVKRRLFPQITLHILPPTQVAMPDAPRARDRRKIAGEMLHQIMMEARMAVRPRETLYESLLSAMYRFGAGKKCVEDVNFTPDSYRKLLTK",
    "reverse_template": "[C:1]-[O;H0;D2;+0:2]-[C;H0;D3;+0:5](-[C;D1;H3:6])=[O;D1;H0:7].[C:3]-[SH;D1;+0:4]>>[C:1]-[OH;D1;+0:2].[C:3]-[S;H0;D2;+0:4]-[C;H0;D3;+0:5](-[C;D1;H3:6])=[O;D1;H0:7]",
    "n_seq": "279",
    "rxnmapper_template": "[C:1]-[O;H0;D2;+0:2]-[C;H0;D3;+0:5](-[C;D1;H3:6])=[O;D1;H0:7].[C:3]-[SH;D1;+0:4]>>[C:1]-[OH;D1;+0:2].[C:3]-[S;H0;D2;+0:4]-[C;H0;D3;+0:5](-[C;D1;H3:6])=[O;D1;H0:7]",
    "localmapper_template": "[C:1]-[O;H0;D2;+0:2]-[C;H0;D3;+0:5](-[C;D1;H3:6])=[O;D1;H0:7].[C:3]-[SH;D1;+0:4]>>[C:1]-[OH;D1;+0:2].[C:3]-[S;H0;D2;+0:4]-[C;H0;D3;+0:5](-[C;D1;H3:6])=[O;D1;H0:7]",
    "Label": "0",
    "similar_rxn": "*N[C@@H](COP(=O)(O)OCC(C)(C)[C@@H](O)C(=O)NCCC(=O)NCCSC(*)=O)C(*)=O.[1*][C@@H](O)CC(=O)N[C@@H](CCCN)C(=O)O>>*C(=O)O[C@H](*)CC(=O)N[C@@H](CCCN)C(=O)O.*N[C@@H](COP(=O)(O)OCC(C)(C)[C@@H](O)C(=O)NCCC(=O)NCCS)C(*)=O",
    "rxn_similarity": "0.6792608153187697",
    "n_enzymes": "4.0"
  }
]
```

### `external_repos/EnzymeCAGE/dataset/internal-test-set/Orphan-335/Orphan-335.csv`
- size_bytes: 607975
- n_rows: 335
- columns: RHEA_ID, DIRECTION, MASTER_ID, UniprotID, SMILES, EC number, CANO_RXN_SMILES, sequence, reverse_template, n_seq, rxnmapper_template, localmapper_template, similar_rxn, rxn_similarity, n_enzymes

```json
[
  {
    "RHEA_ID": "11932",
    "DIRECTION": "UN",
    "MASTER_ID": "11932",
    "UniprotID": "P52704",
    "SMILES": "CC(C)(O)C#N>>CC(C)=O.C#N",
    "EC number": "",
    "CANO_RXN_SMILES": "CC(C)(O)C#N>>C#N.CC(C)=O",
    "sequence": "MAFAHFVLIHTICHGAWIWHKLKPLLEALGHKVTALDLAASGVDPRQIEEIGSFDEYSEPLLTFLEALPPGEKVILVGESCGGLNIAIAADKYCEKIAAAVFHNSVLPDTEHCPSYVVDKLMEVFPDWKDTTYFTYTKDGKEITGLKLGFTLLRENLYTLCGPEEYELAKMLTRKGSLFQNILAKRPFFTKEGYGSIKKIYVWTDQDEIFLPEFQLWQIENYKPDKVYKVEGGDHKLQLT",
    "reverse_template": "[C;D1;H3:1]-[C;H0;D3;+0:2](-[C;D1;H3:3])=[O;H0;D1;+0:4].[CH;D1;+0:5]#[N;D1;H0:6]>>[C;D1;H3:1]-[C;H0;D4;+0:2](-[C;D1;H3:3])(-[OH;D1;+0:4])-[C;H0;D2;+0:5]#[N;D1;H0:6]",
    "n_seq": "257",
    "rxnmapper_template": "[C;D1;H3:1]-[C;H0;D3;+0:2](-[C;D1;H3:3])=[O;H0;D1;+0:4].[CH;D1;+0:5]#[N;D1;H0:6]>>[C;D1;H3:1]-[C;H0;D4;+0:2](-[C;D1;H3:3])(-[OH;D1;+0:4])-[C;H0;D2;+0:5]#[N;D1;H0:6]",
    "localmapper_template": "[C;D1;H3:1]-[C;H0;D3;+0:2](-[C;D1;H3:3])=[O;H0;D1;+0:4].[CH;D1;+0:5]#[N;D1;H0:6]>>[C;D1;H3:1]-[C;H0;D4;+0:2](-[C;D1;H3:3])(-[OH;D1;+0:4])-[C;H0;D2;+0:5]#[N;D1;H0:6]",
    "similar_rxn": "CC(C)/C=N/O.Cc1cc2c(cc1C)N(C[C@H](O)[C@H](O)[C@H](O)COP(=O)(O)O)c1[nH]c(=O)[nH]c(=O)c1N2.O=O>>CC(C)(O)C#N.Cc1cc2nc3c(=O)[nH]c(=O)nc-3n(C[C@H](O)[C@H](O)[C@H](O)COP(=O)(O)O)c2cc1C.O.O",
    "rxn_similarity": "0.7079901129253148",
    "n_enzymes": "4"
  },
  {
    "RHEA_ID": "13105",
    "DIRECTION": "UN",
    "MASTER_ID": "13105",
    "UniprotID": "O14732",
    "SMILES": "O=P([O-])([O-])OC(CO)CO.[H]O[H]>>OCC(O)CO.O=P([O-])([O-])O",
    "EC number": "3.1.3.19",
    "CANO_RXN_SMILES": "O.O=P(O)(O)OC(CO)CO>>O=P(O)(O)O.OCC(O)CO",
    "sequence": "MKPSGEDQAALAAGPWEECFQAAVQLALRAGQIIRKALTEEKRVSTKTSAADLVTETDHLVEDLIISELRERFPSHRFIAEEAAASGAKCVLTHSPTWIIDPIDGTCNFVHRFPTVAVSIGFAVRQELEFGVIYHCTEERLYTGRRGRGAFCNGQRLRVSGETDLSKALVLTEIGPKRDPATLKLFLSNMERLLHAKAHGVRVIGSSTLALCHLASGAADAYYQFGLHCWDLAAATVIIR",
    "reverse_template": "[C:1]-[OH;D1;+0:2].[O;D1;H1:4]-[P;H0;D4;+0:3](-[O;D1;H1:5])(=[O;H0;D1;+0:7])-[OH;D1;+0:6]>>[C:1]-[O;H0;D2;+0:2]-[P;H0;D4;+0:3](-[O;D1;H1:4])(-[O;D1;H1:5])=[O;H0;D1;+0:6].[OH2;D0;+0:7]",
    "n_seq": "288",
    "rxnmapper_template": "[C:1]-[OH;D1;+0:2].[O;D1;H1:4]-[P;H0;D4;+0:3](-[O;D1;H1:5])(=[O;H0;D1;+0:7])-[OH;D1;+0:6]>>[C:1]-[O;H0;D2;+0:2]-[P;H0;D4;+0:3](-[O;D1;H1:4])(-[O;D1;H1:5])=[O;H0;D1;+0:6].[OH2;D0;+0:7]",
    "localmapper_template": "[C:1]-[OH;D1;+0:2].[O;D1;H0:4]=[P;H0;D4;+0:3](-[O;D1;H1:5])(-[O;D1;H1:6])-[OH;D1;+0:7]>>[C:1]-[O;H0;D2;+0:2]-[P;H0;D4;+0:3](=[O;D1;H0:4])(-[O;D1;H1:5])-[O;D1;H1:6].[OH2;D0;+0:7]",
    "similar_rxn": "O.O=P(O)(O)OC[C@@H](O)CO>>O=P(O)(O)O.OCC(O)CO",
    "rxn_similarity": "0.8674236696762899",
    "n_enzymes": "18"
  },
  {
    "RHEA_ID": "14505",
    "DIRECTION": "UN",
    "MASTER_ID": "14505",
    "UniprotID": "Q8NFU3",
    "SMILES": "[H]SS(=O)(=O)[O-].[NH3+][C@@H](CCC(=O)N[C@@H](CS)C(=O)NCC(=O)[O-])C(=O)[O-].[NH3+][C@@H](CCC(=O)N[C@@H](CS)C(=O)NCC(=O)[O-])C(=O)[O-]>>[NH3+][C@@H](CCC(=O)N[C@@H](CSSC[C@H](NC(=O)CC[C@H]([NH3+])C(=O)[O-])C(=O)NCC(=O)[O-])C(=O)NCC(=O)[O-])C(",
    "EC number": "2.8.1.3",
    "CANO_RXN_SMILES": "N[C@@H](CCC(=O)N[C@@H](CS)C(=O)NCC(=O)O)C(=O)O.N[C@@H](CCC(=O)N[C@@H](CS)C(=O)NCC(=O)O)C(=O)O.O=S(=O)(O)S>>N[C@@H](CCC(=O)N[C@@H](CSSC[C@H](NC(=O)CC[C@H](N)C(=O)O)C(=O)NCC(=O)O)C(=O)NCC(=O)O)C(=O)O.O=S(O)O.S",
    "sequence": "MAGAPTVSLPELRSLLASGRARLFDVRSREEAAAGTIPGALNIPVSELESALQMEPAAFQALYSAEKPKLEDEHLVFFCQMGKRGLQATQLARSLGYTGARNYAGAYREWLEKES",
    "reverse_template": "[C:3]-[S;H0;D2;+0:4]-[S;H0;D2;+0:2]-[C:1].[O;D1;H0:5]=[S;H0;D3;+0:6](-[O;D1;H1:7])-[OH;D1;+0:8].[SH2;D0;+0:9]>>[C:1]-[SH;D1;+0:2].[C:3]-[SH;D1;+0:4].[O;D1;H0:5]=[S;H0;D4;+0:6](-[O;D1;H1:7])(=[O;H0;D1;+0:8])-[SH;D1;+0:9]",
    "n_seq": "115",
    "rxnmapper_template": "[C:3]-[S;H0;D2;+0:4]-[S;H0;D2;+0:2]-[C:1].[O;D1;H0:5]=[S;H0;D3;+0:6](-[O;D1;H1:7])-[OH;D1;+0:8].[SH2;D0;+0:9]>>[C:1]-[SH;D1;+0:2].[C:3]-[SH;D1;+0:4].[O;D1;H0:5]=[S;H0;D4;+0:6](-[O;D1;H1:7])(=[O;H0;D1;+0:8])-[SH;D1;+0:9]",
    "localmapper_template": "[C:3]-[S;H0;D2;+0:4]-[S;H0;D2;+0:2]-[C:1].[O;D1;H0:5]=[S;H0;D3;+0:6](-[O;D1;H1:7])-[OH;D1;+0:8].[SH2;D0;+0:9]>>[C:1]-[SH;D1;+0:2].[C:3]-[SH;D1;+0:4].[O;D1;H0:5]=[S;H0;D4;+0:6](-[O;D1;H1:7])(=[O;H0;D1;+0:8])-[SH;D1;+0:9]",
    "similar_rxn": "N[C@@H](CCC(=O)N[C@@H](CS)C(=O)NCC(=O)O)C(=O)O.O=S(=O)(O)S>>N[C@@H](CCC(=O)N[C@@H](CSS)C(=O)NCC(=O)O)C(=O)O.O=S(O)O",
    "rxn_similarity": "0.7719062196281978",
    "n_enzymes": "1"
  }
]
```

