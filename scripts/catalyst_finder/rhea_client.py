from __future__ import annotations

import csv
import io
import re
import threading
import time
from dataclasses import dataclass
from http import HTTPStatus
from pathlib import Path
from typing import Any

import requests

from scripts.catalyst_finder.errors import AppError

RHEA_SEARCH_URL = "https://www.rhea-db.org/rhea/"
RHEA_ENTRY_BASE = "https://www.rhea-db.org/rhea/"
RHEA_SMILES_URL = "https://ftp.expasy.org/databases/rhea/tsv/rhea-reaction-smiles.tsv"
RHEA_DIRECTIONS_URL = "https://ftp.expasy.org/databases/rhea/tsv/rhea-directions.tsv"
USER_AGENT = "NJU-iGEM-2026-CatalystFinder/1.0"
RHEA_ID_RE = re.compile(r"(?:RHEA\s*:\s*)?(\d{5})", re.IGNORECASE)


def canonical_rhea_id(value: str) -> str:
    match = RHEA_ID_RE.search(str(value or ""))
    if not match:
        raise AppError(
            "invalid_rhea_id",
            "请输入有效的 RHEA ID，例如 RHEA:33983。",
            HTTPStatus.UNPROCESSABLE_ENTITY,
        )
    return f"RHEA:{match.group(1)}"


@dataclass(frozen=True)
class RheaCandidate:
    rhea_id: str
    equation: str
    chebi_names: list[str]
    chebi_ids: list[str]
    enzyme_count: int | None
    url: str
    orientation: str = "forward"
    match_score: float = 0.0
    hit_count: int = 0

    def as_dict(self, *, model_ready: bool) -> dict[str, Any]:
        return {
            "rhea_id": self.rhea_id,
            "equation": self.equation,
            "chebi_names": self.chebi_names,
            "chebi_ids": self.chebi_ids,
            "enzyme_count": self.enzyme_count,
            "url": self.url,
            "orientation": self.orientation,
            "model_ready": model_ready,
        }


class RheaClient:
    """Rhea HTTP lookup plus cached direction/Reaction-SMILES reference access."""

    def __init__(self, cache_root: Path, *, user_agent: str = USER_AGENT) -> None:
        self.cache_root = Path(cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})
        self._smiles_lock = threading.Lock()
        self._smiles_by_id: dict[str, str] | None = None
        self._direction_rows: dict[str, dict[str, str]] | None = None

    def search(self, query: str, *, limit: int = 12) -> list[RheaCandidate]:
        query = str(query or "").strip()
        if not query:
            return []
        try:
            response = self.session.get(
                RHEA_SEARCH_URL,
                params={
                    "query": query,
                    "columns": "rhea-id,equation,chebi,chebi-id,uniprot",
                    "format": "tsv",
                    "limit": max(1, min(int(limit), 30)),
                },
                timeout=20,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise AppError(
                "rhea_unavailable",
                "暂时无法连接 Rhea，请稍后重试。",
                HTTPStatus.BAD_GATEWAY,
                str(exc),
            ) from exc
        rows = csv.DictReader(io.StringIO(response.text), delimiter="\t")
        candidates: list[RheaCandidate] = []
        for row in rows:
            raw_id = (row.get("Reaction identifier") or "").strip()
            match = RHEA_ID_RE.search(raw_id)
            if not match:
                continue
            rid = f"RHEA:{match.group(1)}"
            enzymes = (row.get("Enzymes") or "").strip()
            candidates.append(
                RheaCandidate(
                    rhea_id=rid,
                    equation=(row.get("Equation") or "").strip(),
                    chebi_names=[
                        value.strip()
                        for value in (row.get("ChEBI name") or "").split(";")
                        if value.strip()
                    ],
                    chebi_ids=[
                        value.strip()
                        for value in (row.get("ChEBI identifier") or "").split(";")
                        if value.strip()
                    ],
                    enzyme_count=int(enzymes) if enzymes.isdigit() else None,
                    url=f"{RHEA_ENTRY_BASE}{match.group(1)}",
                )
            )
        return candidates

    def exact(self, rhea_id: str) -> RheaCandidate:
        rid = canonical_rhea_id(rhea_id)
        rows = self.search(f"rhea:{rid.split(':', 1)[1]}", limit=5)
        for row in rows:
            if row.rhea_id == rid:
                return row
        raise AppError("rhea_not_found", f"Rhea 中没有找到 {rid}。", HTTPStatus.NOT_FOUND)

    def _ensure_reference_files(self) -> None:
        with self._smiles_lock:
            if self._smiles_by_id is not None and self._direction_rows is not None:
                return
            smiles_path = self.cache_root / "rhea-reaction-smiles.tsv"
            directions_path = self.cache_root / "rhea-directions.tsv"
            self._download_if_needed(RHEA_SMILES_URL, smiles_path, max_age_days=14)
            self._download_if_needed(RHEA_DIRECTIONS_URL, directions_path, max_age_days=14)

            smiles_by_id: dict[str, str] = {}
            with smiles_path.open(encoding="utf-8") as handle:
                for line in handle:
                    line = line.rstrip("\n")
                    if not line or "\t" not in line:
                        continue
                    rid, smiles = line.split("\t", 1)
                    if rid.isdigit() and smiles:
                        smiles_by_id[rid] = smiles

            direction_rows: dict[str, dict[str, str]] = {}
            with directions_path.open(encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle, delimiter="\t"):
                    normalized = {
                        "master": (row.get("RHEA_ID_MASTER") or "").strip(),
                        "lr": (row.get("RHEA_ID_LR") or "").strip(),
                        "rl": (row.get("RHEA_ID_RL") or "").strip(),
                        "bi": (row.get("RHEA_ID_BI") or "").strip(),
                    }
                    for value in normalized.values():
                        if value:
                            direction_rows[value] = normalized
            self._smiles_by_id = smiles_by_id
            self._direction_rows = direction_rows

    def _download_if_needed(self, url: str, path: Path, *, max_age_days: int) -> None:
        if path.is_file() and (time.time() - path.stat().st_mtime) < max_age_days * 86400:
            return
        tmp = path.with_suffix(path.suffix + ".tmp")
        try:
            with self.session.get(url, timeout=45, stream=True) as response:
                response.raise_for_status()
                with tmp.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            tmp.replace(path)
        except requests.RequestException as exc:
            tmp.unlink(missing_ok=True)
            if path.is_file():
                return
            raise AppError(
                "rhea_reference_unavailable",
                "无法取得 Rhea 的标准 Reaction SMILES 数据。",
                HTTPStatus.BAD_GATEWAY,
                str(exc),
            ) from exc

    def reaction_smiles(self, rhea_id: str, orientation: str = "forward") -> dict[str, str]:
        self._ensure_reference_files()
        assert self._smiles_by_id is not None
        assert self._direction_rows is not None
        rid = canonical_rhea_id(rhea_id).split(":", 1)[1]
        if rid in self._smiles_by_id:
            return {
                "source_rhea_id": f"RHEA:{rid}",
                "reaction_smiles": self._smiles_by_id[rid],
            }
        row = self._direction_rows.get(rid)
        if not row:
            raise AppError(
                "rhea_smiles_missing",
                f"{rhea_id} 没有可用的 Rhea Reaction SMILES。",
                HTTPStatus.UNPROCESSABLE_ENTITY,
            )
        preferred = row["rl"] if orientation == "reverse" else row["lr"]
        fallback = row["lr"] or row["rl"]
        chosen = preferred or fallback
        smiles = self._smiles_by_id.get(chosen)
        if not smiles:
            raise AppError(
                "rhea_smiles_missing",
                f"{rhea_id} 没有可用的定向 Reaction SMILES。",
                HTTPStatus.UNPROCESSABLE_ENTITY,
            )
        return {"source_rhea_id": f"RHEA:{chosen}", "reaction_smiles": smiles}
