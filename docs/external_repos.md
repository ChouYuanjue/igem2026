# External Repositories

This project references the following external tools and repositories:

| Directory | Repository |
|-----------|------------|
| `external_repos/DeepSurf` | https://github.com/stemylonas/DeepSurf.git |
| `external_repos/EnzymeCAGE` | https://github.com/GENTEL-lab/EnzymeCAGE.git |
| `external_repos/GENzyme` | https://github.com/WillHua127/GENzyme.git |
| `external_repos/ReactZyme` | https://github.com/WillHua127/ReactZyme.git |
| `external_repos/ScanNet` | https://github.com/jertubiana/ScanNet.git |
| `external_repos/alphafill` | https://github.com/PDB-REDO/alphafill.git |
| `external_repos/fpocket` | https://github.com/Discngine/fpocket.git |
| `external_repos/masif` | https://github.com/LPDI-EPFL/masif.git |
| `external_repos/p2rank` | https://github.com/rdk/p2rank.git |
| `external_repos/igem_database` | https://github.com/Yifan-Jia123/igem_database.git |


## Pinned `igem_database` reference

`igem_database` is managed more strictly than the older unpinned references because
it is used as the database/frontend design baseline. The parent repository tracks:

```text
reproducibility/external_repos/igem_database.lock.json
scripts/setup/sync_igem_database_reference.sh
```

The nested repository itself remains ignored and read-only. It is checked out at a
detached, shallow, partial-clone commit with sparse checkout excluding upstream
`node_modules`.

```bash
bash scripts/setup/sync_igem_database_reference.sh
bash scripts/setup/sync_igem_database_reference.sh --verify-only
```

Do not edit the nested worktree. Put unified frontend code, adapters, and patches in
this repository. See:

- `docs/igem_database_frontend_audit_20260805_zh.md`
- `docs/terpene_atlas_navigator_unified_frontend_plan_20260805_zh.md`
