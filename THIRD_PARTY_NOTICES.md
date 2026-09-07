# Third-party notices and project license status

This repository does **not currently declare a repository-wide license for the project-authored BiME-Rank code and release materials**. Do not infer a project-wide license from a third-party file bundled below. A project-level license should be chosen explicitly by the project owners before claiming that this repository is open-source under a particular license.

Third-party components and external model/data assets remain governed by their own upstream terms. The machine-readable version/checksum boundary is `reproducibility/research_release_manifest.json`.

## Vendored research components

The following small source components are shipped in `projects/active/terpene_screening/third_party/` together with their upstream notices and license texts:

- FusionBench Fisher/RegMean helpers — MIT (`FUSIONBENCH_NOTICE.md`, `FUSIONBENCH_LICENSE`).
- Mammoth soft-target distillation primitive — MIT (`MAMMOTH_NOTICE.md`, `MAMMOTH_LICENSE`).
- neural-ranking-kd MarginMSE loss — Apache-2.0 (`NEURAL_RANKING_KD_NOTICE.md`, `NEURAL_RANKING_KD_LICENSE`).
- RecAdam optimizer — Apache-2.0 (`RECAdam_NOTICE.md`, `RECAdam_LICENSE`).
- TIES-Merging primitives — BSD-3-Clause (`TIES_NOTICE.md`, `TIES_LICENSE`).

Those licenses apply to the corresponding upstream-derived files, not automatically to the rest of this repository.

## External model and method dependencies

Large third-party repositories/checkpoints are not vendored as ordinary Git assets. Before downloading or using them, review the exact upstream license at the pinned source/version recorded in the release manifest.

- **CLIPZyme** — the locally pinned upstream repository carries Apache License 2.0 (`LICENSE.txt` at commit `6e48ae05e2cf705af16368afc579246d80767326`).
- **Horizyn** — the pinned upstream repository carries PolyForm Noncommercial License 1.0.0 (commit `e6655e732f574c8bfa0488b9bc5068b67e382745`).
- **EnzymeCAGE** — the locally pinned upstream `LICENSE` is non-commercial and also contains additional explicit use restrictions. The exact upstream license text is authoritative; do not treat the method/checkpoints as generally unrestricted research software.
- **EnzGFM** — no license file was present in the pinned local model/reference copy used for this release audit. The model remains an external dependency; users must verify the upstream terms before redistribution or use.
- **RDKitPlus-derived feature route** — project metadata records it as an external PolyForm Noncommercial 1.0.0 research dependency; its source is not copied into this repository.

This notice is an inventory aid, not a substitute for the upstream license texts or legal advice.
