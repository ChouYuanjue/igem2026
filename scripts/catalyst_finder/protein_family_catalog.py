from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

PFAM_RE = re.compile(r"\bPF\d{5}\b", re.IGNORECASE)


def _norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).strip()


@dataclass(frozen=True)
class ProteinFamily:
    family_id: str
    label: str
    member_ids: tuple[str, ...]
    source: str
    query_scope: str
    aliases: tuple[str, ...] = ()
    caution: str = ""
    caution_zh: str = ""
    scope_note: str = ""
    scope_note_zh: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "family_id": self.family_id,
            "label": self.label,
            "member_count": len(self.member_ids),
            "member_ids_sample": list(self.member_ids[:12]),
            "source": self.source,
            "query_scope": self.query_scope,
            "caution": self.caution,
            "caution_zh": self.caution_zh,
            "scope_note": self.scope_note,
            "scope_note_zh": self.scope_note_zh,
        }


# These aliases encode biological nomenclature, not model truth. They only map a
# user-facing family name to a locally auditable family definition. Catalytic
# activity is still derived independently from database association evidence.
CURATED_FAMILY_ALIASES: dict[str, dict[str, object]] = {
    "PF01040": {
        "label": "UbiA prenyltransferase family (PF01040)",
        "aliases": (
            "ubia",
            "ubia type",
            "ubia-type",
            "ubia prenyltransferase",
            "ubia-type prenyltransferase",
            "ubia type terpene cyclase",
            "ubia-type terpene cyclase",
            "ubia型萜环化酶",
            "ubia 型萜环化酶",
        ),
        "caution": (
            "PF01040 is broader than experimentally validated UbiA-type terpene "
            "cyclases. Family membership is a scope definition, not catalytic validation."
        ),
        "caution_zh": (
            "PF01040 的范围比已经实验验证的 UbiA 型萜环化酶更宽；"
            "家族成员身份只用于界定查询范围，不代表该成员已被证明具有萜环化活性。"
        ),
    },
}


class ProteinFamilyCatalog:
    """Resolve protein family/class queries against locally auditable memberships.

    Membership is assembled from the merged candidate metadata plus the project's
    Pfam annotation snapshot. The catalog never turns family membership into a
    positive enzyme↔reaction fact; callers must query the evidence layer separately.
    """

    def __init__(self, root: Path, candidate_ids: Iterable[str]) -> None:
        self.root = Path(root).resolve()
        self.candidate_ids = {
            str(value).strip().upper()
            for value in candidate_ids
            if str(value).strip()
        }
        self._loaded = False
        self._pfam_members: dict[str, set[str]] = {}
        self._domain_members: dict[str, set[str]] = {}

    def _add_pfam(self, protein_id: str, pfam: str) -> None:
        protein = str(protein_id or "").strip().upper()
        family = str(pfam or "").strip().upper()
        if (
            not protein
            or protein not in self.candidate_ids
            or not PFAM_RE.fullmatch(family)
        ):
            return
        self._pfam_members.setdefault(family, set()).add(protein)

    def _load(self) -> None:
        if self._loaded:
            return
        metadata = (
            self.root
            / "data/catalyst_candidate_universes/general_merged/protein_metadata.csv"
        )
        if metadata.is_file():
            with metadata.open(encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    protein = str(row.get("protein_id") or "").strip().upper()
                    for token in re.split(r"[;,\s]+", str(row.get("pfam") or "")):
                        if token:
                            self._add_pfam(protein, token)
                    domain = str(row.get("domain_family") or "").strip()
                    if protein in self.candidate_ids and domain:
                        self._domain_members.setdefault(_norm(domain), set()).add(protein)

        project_pfam = (
            self.root
            / "data/terpene_current_pfam_uniprot_v2/current_pfam_groups.csv"
        )
        if project_pfam.is_file():
            with project_pfam.open(encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    protein = str(row.get("Entry") or "").strip().upper()
                    for token in str(row.get("pfam_combination") or "").split(";"):
                        self._add_pfam(protein, token)
        self._loaded = True

    def family(self, family_id: str) -> ProteinFamily | None:
        self._load()
        value = str(family_id or "").strip()
        upper = value.upper()
        if PFAM_RE.fullmatch(upper):
            members = tuple(sorted(self._pfam_members.get(upper, ())))
            if not members:
                return None
            meta = CURATED_FAMILY_ALIASES.get(upper, {})
            return ProteinFamily(
                family_id=upper,
                label=str(meta.get("label") or f"Pfam {upper}"),
                member_ids=members,
                source="project_pfam_snapshot+general_merged_metadata",
                query_scope="locally_annotated_candidate_subset",
                aliases=tuple(str(x) for x in meta.get("aliases", ())),
                caution=str(
                    meta.get("caution")
                    or "Pfam membership defines sequence-family scope; it is not catalytic validation."
                ),
                caution_zh=str(
                    meta.get("caution_zh")
                    or "Pfam 家族成员身份只用于界定序列家族范围，不代表催化活性已经得到验证。"
                ),
                scope_note=(
                    "Members shown here are the intersection of locally available Pfam "
                    "annotations and the active general candidate universe, not the complete "
                    "UniProt/Pfam family."
                ),
                scope_note_zh=(
                    "这里的成员是当前本地可用 Pfam 标注与通用候选库的交集，"
                    "不代表 UniProt/Pfam 中该家族的全部成员。"
                ),
            )
        key = _norm(value)
        members = tuple(sorted(self._domain_members.get(key, ())))
        if members:
            return ProteinFamily(
                family_id=f"DOMAIN:{key.replace(' ', '_')}",
                label=value.replace("_", " "),
                member_ids=members,
                source="general_merged_metadata",
                query_scope="domain_family",
                caution=(
                    "Domain-family membership defines candidate scope; it is not catalytic validation."
                ),
                caution_zh=(
                    "结构域家族成员身份只用于界定候选范围，不代表催化活性已经得到验证。"
                ),
                scope_note=(
                    "Members shown here are limited to the active general candidate universe "
                    "with matching local metadata."
                ),
                scope_note_zh=(
                    "这里仅展示当前通用候选库中具有相应本地元数据标注的成员。"
                ),
            )
        return None

    def resolve(self, *values: str) -> ProteinFamily | None:
        self._load()
        texts = [
            str(value or "").strip()
            for value in values
            if str(value or "").strip()
        ]
        joined = " ".join(texts)
        explicit = PFAM_RE.search(joined)
        if explicit:
            family = self.family(explicit.group(0).upper())
            if family:
                return family

        normalized = _norm(joined)
        for family_id, meta in CURATED_FAMILY_ALIASES.items():
            aliases = tuple(str(value) for value in meta.get("aliases", ()))
            for alias in aliases:
                key = _norm(alias)
                if key and (
                    normalized == key
                    or re.search(rf"(?:^| ){re.escape(key)}(?: |$)", normalized)
                ):
                    family = self.family(family_id)
                    if family:
                        return family

        for domain_key in sorted(self._domain_members, key=len, reverse=True):
            if domain_key and (
                normalized == domain_key
                or re.search(rf"(?:^| ){re.escape(domain_key)}(?: |$)", normalized)
            ):
                return self.family(domain_key)
        return None
