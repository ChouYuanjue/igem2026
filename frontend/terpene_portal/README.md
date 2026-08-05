# TerpeneNavigator model portal

This frontend belongs to the parent `igem2026` repository. It does not modify
`external_repos/igem_database`.

## Product boundary

The portal has two top-level views:

- **Model Navigator**: the parent project's production Reaction→Enzyme and
  Enzyme→Reaction retrieval workflow. Its SVG execution graph lights the actual
  route stage-by-stage, shows active/skipped retrieval lanes, RRF or direct
  fusion, the locked rank, applicability, conformal coverage, Evidence
  Passports, candidate inspection, and CSV/JSON export.
- **Database Atlas** contains two read-only surfaces. **Model Data Hub** reads
  the deployed 2,085-protein / 753-reaction / 3,439-association universe through
  a parent-repository adapter. **Upstream Atlas** serves the database team's
  already-built frontend from `external_repos/igem_database/frontend/dist`.

The database bridge only supports read operations needed by upstream features
that already exist: map loading, entry search, pathway search, overlapping-edge
expansion, enzyme/reaction/compound detail, and structure/atom-map assets. It
explicitly rejects writes and does not implement unfinished database features.

## Build on the model server

```bash
bash scripts/terpene_portal/build.sh
```

The build script uses Node from `PATH` when available. Otherwise it downloads a
portable Node runtime into the user's cache, outside the repository.

## Deploy

```bash
bash scripts/terpene_portal/manage.sh start
bash scripts/terpene_portal/manage.sh status
bash scripts/terpene_portal/manage.sh logs
bash scripts/terpene_portal/manage.sh stop
```

Default URL:

```text
http://<model-server>:8787/portal/
```

The raw upstream database frontend is mounted at `/database/`. The retrieval
API is mounted under `/api/model/`; the upstream-compatible read-only bridge
retains `/api/v1/`. Model Data Hub uses these parent-owned, read-only endpoints:

```text
/api/model-data/summary
/api/model-data/search?q=<query>&kind=<all|protein|reaction|association>
/api/model-data/graph?q=<query>&focus_id=<entity-id>&limit=<n>
/api/model-data/entities/<protein|reaction>/<entity-id>
```

No model-data endpoint accepts writes, and no file inside the nested database
repository is changed.

To proxy a live database backend instead of the compatibility snapshot:

```bash
TERPENE_DATABASE_API_URL=http://127.0.0.1:8000 \
  bash scripts/terpene_portal/manage.sh restart
```

Only the allowlisted read endpoints are proxied.
