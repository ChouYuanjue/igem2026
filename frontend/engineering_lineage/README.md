# BRIDGE Engineering Atlas

This directory contains the static interactive visualization for the BRIDGE engineering lineage.

Public route:

`https://nju-igem.runnelzhang.com/engineering/`

The page deliberately starts at **EnzymeCAGE**, then follows the problem-definition lineage through candidate expansion, Broad Retrieval, BiME-Rank, the FIBRE detour, query-level expert applicability, specialist return, and BRIDGE.

The page is served by an isolated local service on `127.0.0.1:8866`. It does not share the COMPASS runtime process or database frontend process.

Repository service definition:

`scripts/engineering_lineage/engineering-lineage.service`

The `nju-igem` Cloudflare tunnel routes `/engineering` to this service before the existing COMPASS catch-all rule.
