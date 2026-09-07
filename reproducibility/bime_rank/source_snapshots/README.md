# Reproducibility source snapshots

Exact copies of formerly ignored one-off scripts. Original locations and hashes are in manifest.json. These are historical execution recipes, not new production entrypoints. Restore to the original path before replaying: some scripts resolve files relative to their location. Running them may start expensive computation; indexing and verification never run them.
Recovered canonical evaluation entrypoints include the exact CLIPZyme R2E strict external confirmation, CLIPZyme E2R strict fair comparison, and reciprocal-consistency external confirmation evaluators. These were present in ignored `results/` directories on the development server and are now tracked byte-for-byte here; no experiment rerun was needed to recover them.
