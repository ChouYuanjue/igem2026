# External Repositories

This directory is for read-only third-party repositories used as dependencies
or research references.

Do not edit files inside this directory. If an adapter, runner, analysis script,
or patch experiment is needed, place it under `projects/`, `scripts/`, or
`docs/` in this repository.

Use `scripts/setup/clone_external_repos.sh` to collect the external repositories used
by the first exploration phase.


## Pinned design reference

`igem_database/` is a pinned nested Git worktree used to audit and reference the
terpene database frontend. Its exact commit and sparse-checkout contract are stored
in `reproducibility/external_repos/igem_database.lock.json`. Use
`scripts/setup/sync_igem_database_reference.sh`; never edit the nested repository
or install packages inside it as part of the parent repository workflow.
