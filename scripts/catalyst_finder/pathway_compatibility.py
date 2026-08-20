from __future__ import annotations

import hashlib
import json
import math
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote

import requests

try:
    from Bio.SeqUtils.ProtParam import ProteinAnalysis
except ImportError:  # optional: the main compatibility workflow still runs without it
    ProteinAnalysis = None

UNIPROT_ENTRY_URL = "https://rest.uniprot.org/uniprotkb/{accession}.json"


def _clean_texts(block: Any) -> list[str]:
    if not isinstance(block, dict):
        return []
    values: list[str] = []
    for row in block.get("texts") or []:
        if isinstance(row, dict):
            value = str(row.get("value") or "").strip()
            if value:
                values.append(value)
    return values


def _range(a: Any, b: Any | None = None) -> list[float] | None:
    try:
        left = float(a)
        right = float(a if b is None else b)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(left) and math.isfinite(right)):
        return None
    return [min(left, right), max(left, right)]


def _first_range(patterns: list[re.Pattern[str]], text: str) -> list[float] | None:
    for pattern in patterns:
        match = pattern.search(text)
        if not match:
            continue
        result = _range(match.group(1), match.group(2) if match.lastindex and match.lastindex >= 2 else None)
        if result:
            return result
    return None


PH_OPTIMUM_PATTERNS = [
    re.compile(r"optimum\s+pH\s+(?:is|of[^.]{0,24}?is)\s+(?:between\s+)?([0-9]+(?:\.[0-9]+)?)\s*(?:-|–|to|and)\s*([0-9]+(?:\.[0-9]+)?)", re.I),
    re.compile(r"optimum\s+pH\s+(?:is|of[^.]{0,24}?is)\s+(?:about\s+|around\s+)?([0-9]+(?:\.[0-9]+)?)", re.I),
]
PH_ACTIVE_PATTERNS = [
    re.compile(r"(?:stable|active|activity[^.]{0,35})\s+(?:at\s+)?(?:from|between|over[^.]{0,12}?range)\s+pH\s*([0-9]+(?:\.[0-9]+)?)\s*(?:-|–|to|and)\s*([0-9]+(?:\.[0-9]+)?)", re.I),
    re.compile(r"(?:stable|active|activity[^.]{0,35})\s+(?:at\s+)?pH\s*(?:range\s*)?([0-9]+(?:\.[0-9]+)?)\s*(?:-|–|to|and)\s*([0-9]+(?:\.[0-9]+)?)", re.I),
    re.compile(r"pH\s+range\s+([0-9]+(?:\.[0-9]+)?)\s*(?:-|–|to)\s*([0-9]+(?:\.[0-9]+)?)", re.I),
]
TEMP_OPTIMUM_PATTERNS = [
    re.compile(r"optimum\s+temperature(?:\s+of[^.]{0,30}?)?\s+(?:is|of[^.]{0,20}?is)\s+(?:between\s+)?([0-9]+(?:\.[0-9]+)?)\s*(?:-|–|to|and)\s*([0-9]+(?:\.[0-9]+)?)\s*(?:degrees?\s+Celsius|°C)", re.I),
    re.compile(r"optimum\s+temperature(?:\s+of[^.]{0,30}?)?\s+(?:is|of[^.]{0,20}?is)\s+(?:about\s+|around\s+)?([0-9]+(?:\.[0-9]+)?)\s*(?:degrees?\s+Celsius|°C)", re.I),
]
TEMP_ACTIVE_PATTERNS = [
    re.compile(r"(?:stable|active)\s+(?:at\s+)?(?:from|between|over[^.]{0,16}?range)\s+([0-9]+(?:\.[0-9]+)?)\s*(?:-|–|to|and)\s*([0-9]+(?:\.[0-9]+)?)\s*(?:degrees?\s+Celsius|°C)", re.I),
    re.compile(r"stable\s+(?:up\s+to|below)\s+([0-9]+(?:\.[0-9]+)?)\s*(?:degrees?\s+Celsius|°C)", re.I),
]


def _mid(interval: list[float] | None) -> float | None:
    if not interval:
        return None
    return (float(interval[0]) + float(interval[1])) / 2.0


def _overlap(a: list[float] | None, b: list[float] | None) -> list[float] | None:
    if not a or not b:
        return None
    left = max(float(a[0]), float(b[0]))
    right = min(float(a[1]), float(b[1]))
    return [left, right] if left <= right else []


def _interval_distance(a: list[float] | None, b: list[float] | None) -> float | None:
    if not a or not b:
        return None
    if _overlap(a, b):
        return 0.0
    if a[1] < b[0]:
        return float(b[0] - a[1])
    return float(a[0] - b[1])


def _intersection(intervals: list[list[float]]) -> list[float] | None:
    if not intervals:
        return None
    left = max(row[0] for row in intervals)
    right = min(row[1] for row in intervals)
    return [left, right] if left <= right else []


def _format_interval(value: list[float] | None, *, suffix: str = "") -> str | None:
    if value is None:
        return None
    if value == []:
        return "无共同区间"
    left, right = value
    if abs(left - right) < 1e-9:
        return f"{left:g}{suffix}"
    return f"{left:g}–{right:g}{suffix}"


class UniProtConditionEvidence:
    def __init__(self, cache_root: Path, *, user_agent: str) -> None:
        self.cache_root = cache_root / "uniprot_conditions"
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})
        self._lock = threading.Lock()

    def _cache_path(self, accession: str) -> Path:
        key = re.sub(r"[^A-Za-z0-9_.-]+", "_", accession.upper())
        return self.cache_root / f"{key}.json"

    def entry(self, accession: str, *, max_age_days: int = 14) -> dict[str, Any]:
        accession = str(accession or "").strip().upper()
        if not accession:
            return {}
        path = self._cache_path(accession)
        if path.exists() and time.time() - path.stat().st_mtime <= max_age_days * 86400:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                return data if isinstance(data, dict) else {}
            except (OSError, json.JSONDecodeError):
                pass
        response = self.session.get(UNIPROT_ENTRY_URL.format(accession=quote(accession, safe="")), timeout=25)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict):
            with self._lock:
                tmp = path.with_suffix(".tmp")
                tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
                tmp.replace(path)
            return data
        return {}

    def profile(self, accession: str) -> dict[str, Any]:
        try:
            row = self.entry(accession)
        except requests.RequestException as exc:
            return {
                "accession": accession,
                "available": False,
                "source": "UniProtKB",
                "error": f"{type(exc).__name__}: {exc}",
            }
        if not row:
            return {"accession": accession, "available": False, "source": "UniProtKB"}
        comments = row.get("comments") or []
        ph_texts: list[str] = []
        temp_texts: list[str] = []
        cofactors: list[str] = []
        cofactor_chebi: list[str] = []
        regulation: list[str] = []
        locations: list[str] = []
        ec_numbers: list[str] = []
        rhea_ids: list[str] = []
        evidence_pubmed: list[str] = []
        for comment in comments:
            if not isinstance(comment, dict):
                continue
            ctype = str(comment.get("commentType") or "")
            if ctype == "BIOPHYSICOCHEMICAL PROPERTIES":
                ph_texts.extend(_clean_texts(comment.get("phDependence")))
                temp_texts.extend(_clean_texts(comment.get("temperatureDependence")))
            elif ctype == "COFACTOR":
                for item in comment.get("cofactors") or []:
                    if not isinstance(item, dict):
                        continue
                    name = str(item.get("name") or "").strip()
                    if name and name not in cofactors:
                        cofactors.append(name)
                    xref = item.get("cofactorCrossReference") or {}
                    if xref.get("database") == "ChEBI" and xref.get("id") and xref.get("id") not in cofactor_chebi:
                        cofactor_chebi.append(str(xref["id"]))
                    for ev in item.get("evidences") or []:
                        if isinstance(ev, dict) and ev.get("source") == "PubMed" and ev.get("id"):
                            evidence_pubmed.append(str(ev["id"]))
            elif ctype == "ACTIVITY REGULATION":
                regulation.extend(_clean_texts(comment))
            elif ctype == "SUBCELLULAR LOCATION":
                for item in comment.get("subcellularLocations") or []:
                    loc = ((item or {}).get("location") or {}).get("value") if isinstance(item, dict) else None
                    if loc and str(loc) not in locations:
                        locations.append(str(loc))
            elif ctype == "CATALYTIC ACTIVITY":
                reaction = comment.get("reaction") or {}
                ec = str(reaction.get("ecNumber") or "").strip()
                if ec and ec not in ec_numbers:
                    ec_numbers.append(ec)
                for xref in reaction.get("reactionCrossReferences") or []:
                    if isinstance(xref, dict) and xref.get("database") == "Rhea" and xref.get("id"):
                        rid = str(xref["id"])
                        if rid not in rhea_ids:
                            rhea_ids.append(rid)
                for ev in reaction.get("evidences") or []:
                    if isinstance(ev, dict) and ev.get("source") == "PubMed" and ev.get("id"):
                        evidence_pubmed.append(str(ev["id"]))
        ph_text = " ".join(ph_texts)
        temp_text = " ".join(temp_texts)
        ph_optimum = _first_range(PH_OPTIMUM_PATTERNS, ph_text)
        ph_active = _first_range(PH_ACTIVE_PATTERNS, ph_text)
        temp_optimum = _first_range(TEMP_OPTIMUM_PATTERNS, temp_text)
        temp_active = _first_range(TEMP_ACTIVE_PATTERNS, temp_text)
        organism = str((row.get("organism") or {}).get("scientificName") or "").strip() or None
        lineage = [str(x) for x in ((row.get("organism") or {}).get("lineage") or []) if str(x).strip()]
        reviewed = "reviewed" in str(row.get("entryType") or "").lower()
        membrane = any("membrane" in loc.casefold() for loc in locations)
        sequence = str((row.get("sequence") or {}).get("value") or "").strip().upper()
        theoretical_pi = None
        if ProteinAnalysis is not None and sequence and re.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]+", sequence):
            try:
                theoretical_pi = round(float(ProteinAnalysis(sequence).isoelectric_point()), 3)
            except (ValueError, ZeroDivisionError):
                theoretical_pi = None
        return {
            "accession": str(row.get("primaryAccession") or accession),
            "available": True,
            "source": "UniProtKB",
            "reviewed": reviewed,
            "organism": organism,
            "lineage": lineage,
            "ph_optimum": ph_optimum,
            "ph_active": ph_active,
            "ph_text": ph_text or None,
            "temperature_optimum_c": temp_optimum,
            "temperature_active_c": temp_active,
            "temperature_text": temp_text or None,
            "cofactors": cofactors,
            "cofactor_chebi": cofactor_chebi,
            "activity_regulation": regulation,
            "locations": locations,
            "membrane_associated": membrane,
            "theoretical_pi": theoretical_pi,
            "sequence_length": len(sequence) if sequence else None,
            "ec_numbers": ec_numbers,
            "rhea_ids": rhea_ids,
            "evidence_pubmed": sorted(set(evidence_pubmed)),
            "url": f"https://www.uniprot.org/uniprotkb/{quote(str(row.get('primaryAccession') or accession), safe='')}",
        }


def _model_utility(candidate: dict[str, Any]) -> float:
    rank = max(1, int(candidate.get("rank") or 1))
    fraction = max(0.0, min(1.0, float(candidate.get("score_fraction") or 0.0)))
    rank_term = 1.0 / (1.0 + 0.16 * (rank - 1))
    return 0.62 * fraction + 0.38 * rank_term


def _coverage(profile: dict[str, Any]) -> int:
    return sum(bool(profile.get(key)) for key in ("ph_optimum", "ph_active", "temperature_optimum_c", "temperature_active_c", "cofactors", "locations", "activity_regulation"))


def _regulation_blob(profile: dict[str, Any]) -> str:
    return " ".join(str(x) for x in profile.get("activity_regulation") or []).casefold()


def _cofactor_key(value: Any) -> str:
    text = str(value or "").strip().casefold()
    aliases = {
        "magnesium": "mg",
        "manganese": "mn",
        "calcium": "ca",
        "zinc": "zn",
        "iron": "fe",
        "ferrous": "fe2",
        "ferric": "fe3",
        "cobalt": "co",
        "copper": "cu",
        "nickel": "ni",
    }
    for word, symbol in aliases.items():
        text = re.sub(rf"\b{word}\b", symbol, text)
    return re.sub(r"[^a-z0-9]+", "", text)


def _point_interval_distance(value: float, interval: list[float] | None) -> float | None:
    if not interval:
        return None
    if interval[0] <= value <= interval[1]:
        return 0.0
    return min(abs(value - interval[0]), abs(value - interval[1]))


def target_condition_compatibility(candidate: dict[str, Any], target: dict[str, Any]) -> tuple[float, list[dict[str, Any]]]:
    profile = candidate.get("condition_profile") or {}
    score = 0.0
    issues: list[dict[str, Any]] = []
    ph = target.get("ph")
    if ph is not None:
        interval = profile.get("ph_active") or profile.get("ph_optimum")
        distance = _point_interval_distance(float(ph), interval)
        if distance is not None:
            if distance == 0:
                score += 0.035
            elif distance >= 1.0:
                severity = "high" if distance >= 2.0 else "medium"
                score -= 0.18 if severity == "high" else 0.09
                issues.append({"severity": severity, "type": "target_ph", "detail": f"用户目标 pH {float(ph):g} 距该酶已报道区间约 {distance:g} 个 pH 单位。"})
    temperature = target.get("temperature_c")
    if temperature is not None:
        interval = profile.get("temperature_active_c") or profile.get("temperature_optimum_c")
        distance = _point_interval_distance(float(temperature), interval)
        if distance is not None:
            if distance == 0:
                score += 0.035
            elif distance >= 10.0:
                severity = "high" if distance >= 20.0 else "medium"
                score -= 0.18 if severity == "high" else 0.09
                issues.append({"severity": severity, "type": "target_temperature", "detail": f"用户目标温度 {float(temperature):g} °C 距该酶已报道区间约 {distance:g} °C。"})
    requested = {_cofactor_key(x) for x in target.get("cofactors") or [] if _cofactor_key(x)}
    known = {_cofactor_key(x) for x in profile.get("cofactors") or [] if _cofactor_key(x)}
    if requested and known and requested & known:
        score += 0.025
    return score, issues


def pairwise_compatibility(a: dict[str, Any], b: dict[str, Any], *, mode: str) -> tuple[float, list[dict[str, Any]]]:
    pa = a.get("condition_profile") or {}
    pb = b.get("condition_profile") or {}
    score = 0.0
    issues: list[dict[str, Any]] = []

    shared_operation = mode != "sequential"
    active_a = pa.get("ph_active") or pa.get("ph_optimum")
    active_b = pb.get("ph_active") or pb.get("ph_optimum")
    ph_distance = _interval_distance(active_a, active_b) if shared_operation else None
    if ph_distance is not None:
        if ph_distance == 0:
            score += 0.035
        elif ph_distance >= 1.5:
            severity = "high" if ph_distance >= 2.5 else "medium"
            score -= 0.20 if severity == "high" else 0.11
            issues.append({"severity": severity, "type": "ph", "detail": f"已报道的 pH 活性/最适区间相距约 {ph_distance:g} 个 pH 单位。"})
    elif shared_operation:
        ma, mb = _mid(pa.get("ph_optimum")), _mid(pb.get("ph_optimum"))
        if ma is not None and mb is not None:
            distance = abs(ma - mb)
            if distance >= 2.0:
                severity = "high" if distance >= 3.0 else "medium"
                score -= 0.18 if severity == "high" else 0.10
                issues.append({"severity": severity, "type": "ph", "detail": f"最适 pH 相差约 {distance:g}。"})

    temp_a = pa.get("temperature_active_c") or pa.get("temperature_optimum_c")
    temp_b = pb.get("temperature_active_c") or pb.get("temperature_optimum_c")
    temp_distance = _interval_distance(temp_a, temp_b) if shared_operation else None
    if temp_distance is not None:
        if temp_distance == 0:
            score += 0.035
        elif temp_distance >= 15:
            severity = "high" if temp_distance >= 25 else "medium"
            score -= 0.20 if severity == "high" else 0.11
            issues.append({"severity": severity, "type": "temperature", "detail": f"已报道温度区间相距约 {temp_distance:g} °C。"})
    elif shared_operation:
        ma, mb = _mid(pa.get("temperature_optimum_c")), _mid(pb.get("temperature_optimum_c"))
        if ma is not None and mb is not None:
            distance = abs(ma - mb)
            if distance >= 20:
                severity = "high" if distance >= 30 else "medium"
                score -= 0.18 if severity == "high" else 0.10
                issues.append({"severity": severity, "type": "temperature", "detail": f"最适温度相差约 {distance:g} °C。"})

    cof_a = {_cofactor_key(x): str(x) for x in pa.get("cofactors") or [] if _cofactor_key(x)}
    cof_b = {_cofactor_key(x): str(x) for x in pb.get("cofactors") or [] if _cofactor_key(x)}
    shared = set(cof_a) & set(cof_b)
    if shared_operation and shared:
        score += min(0.035, 0.018 * len(shared))
    if shared_operation:
        reg_a, reg_b = _regulation_blob(pa), _regulation_blob(pb)
        for key, original in {**cof_a, **cof_b}.items():
            if key and ((key in reg_a and re.search(r"inhibit|inactiv|suppress|abolish", reg_a)) or (key in reg_b and re.search(r"inhibit|inactiv|suppress|abolish", reg_b))):
                score -= 0.12
                issues.append({"severity": "medium", "type": "cofactor_regulation", "detail": f"辅因子/金属 {original} 与另一酶的活性调控注释可能存在干扰，需回查原始文献。"})
                break

    loc_a = {str(x).casefold(): str(x) for x in pa.get("locations") or []}
    loc_b = {str(x).casefold(): str(x) for x in pb.get("locations") or []}
    if mode == "in_vivo" and loc_a and loc_b:
        if set(loc_a) & set(loc_b):
            score += 0.025
        else:
            score -= 0.14
            issues.append({"severity": "medium", "type": "localization", "detail": "两步酶的 UniProt 亚细胞定位注释不一致；若在同一底盘表达，需要检查靶向/区室化设计。"})

    if mode in {"one_pot", "auto"} and (pa.get("membrane_associated") or pb.get("membrane_associated")):
        score -= 0.05
    return score, issues


class PathwayCompatibilityAnalyzer:
    """Global candidate-set compatibility layer on top of the existing R2E ranker.

    The analyzer deliberately does not claim to predict precipitation. It uses
    curated condition annotations as evidence and reports missing evidence as
    uncertainty rather than compatibility.
    """

    def __init__(
        self,
        *,
        root: Path,
        catalog: Any,
        rank_reaction: Callable[..., dict[str, Any]],
        user_agent: str,
        cache_root: Path,
    ) -> None:
        self.root = root
        self.catalog = catalog
        self.rank_reaction = rank_reaction
        self.conditions = UniProtConditionEvidence(cache_root, user_agent=user_agent)

    def _accession(self, candidate: dict[str, Any]) -> str | None:
        value = str(candidate.get("uniprot_id") or candidate.get("accession") or "").strip()
        if value:
            return value
        cid = str(candidate.get("candidate_id") or candidate.get("id") or "").strip()
        meta = self.catalog.protein_by_id.get(cid, {}) if cid else {}
        value = str(meta.get("uniprot_id") or "").strip()
        if value:
            return value
        if re.fullmatch(r"[A-Z0-9]{6}(?:[A-Z0-9]{4})?", cid, re.I):
            return cid.upper()
        return None

    def _enrich_candidates(self, step_candidates: list[list[dict[str, Any]]]) -> None:
        jobs: dict[Any, tuple[int, int, str]] = {}
        with ThreadPoolExecutor(max_workers=6) as pool:
            for si, rows in enumerate(step_candidates):
                for ci, candidate in enumerate(rows):
                    accession = self._accession(candidate)
                    candidate["profile_accession"] = accession
                    if accession:
                        jobs[pool.submit(self.conditions.profile, accession)] = (si, ci, accession)
                    else:
                        candidate["condition_profile"] = {"available": False, "source": "UniProtKB", "accession": None}
            for future in as_completed(jobs):
                si, ci, accession = jobs[future]
                try:
                    profile = future.result()
                except Exception as exc:  # defensive: compatibility evidence must not break ranking
                    profile = {"available": False, "source": "UniProtKB", "accession": accession, "error": f"{type(exc).__name__}: {exc}"}
                step_candidates[si][ci]["condition_profile"] = profile

    def _explicit_candidate(self, enzyme_id: str) -> dict[str, Any]:
        raw = str(enzyme_id or "").strip()
        local = self.catalog.protein_by_id.get(raw, {})
        if local:
            accession = str(local.get("uniprot_id") or raw).strip()
            return {
                "rank": 1,
                "base_rank": 1,
                "candidate_id": raw,
                "score": 1.0,
                "score_fraction": 1.0,
                "uniprot_id": accession,
                "uniprot_url": f"https://www.uniprot.org/uniprotkb/{quote(accession, safe='')}",
                "name": local.get("name") if local.get("name") != raw else None,
                "species": local.get("species"),
                "candidate_source": "user_confirmed",
                "selection_source": "user_confirmed",
                "known_association": False,
                "explicit": True,
            }
        return {
            "rank": 1,
            "base_rank": 1,
            "candidate_id": raw,
            "score": 1.0,
            "score_fraction": 1.0,
            "uniprot_id": raw,
            "uniprot_url": f"https://www.uniprot.org/uniprotkb/{quote(raw, safe='')}",
            "name": None,
            "species": None,
            "candidate_source": "user_confirmed_external",
            "selection_source": "user_confirmed",
            "known_association": False,
            "explicit": True,
        }

    def analyze(self, *, steps: list[dict[str, Any]], user_text: str = "", execution_mode: str = "auto", host: str = "", target_conditions: dict[str, Any] | None = None) -> dict[str, Any]:
        mode = execution_mode if execution_mode in {"auto", "one_pot", "sequential", "in_vivo"} else "auto"
        target = dict(target_conditions or {})
        if len(steps) < 2:
            raise ValueError("pathway analysis requires at least two reaction steps")
        if len(steps) > 8:
            raise ValueError("pathway analysis currently supports at most eight steps")

        resolved_steps: list[dict[str, Any]] = []
        step_candidates: list[list[dict[str, Any]]] = []
        for index, step in enumerate(steps, start=1):
            rid = str(step.get("rhea_id") or "").strip()
            if not rid:
                raise ValueError(f"missing Rhea ID for step {index}")
            orientation = "reverse" if str(step.get("orientation") or "forward") == "reverse" else "forward"
            explicit_enzyme = str(step.get("enzyme_id") or "").strip()
            if explicit_enzyme:
                rows = [self._explicit_candidate(explicit_enzyme)]
                ranking_meta: dict[str, Any] = {"source": "user_confirmed", "candidate_count": 1}
            else:
                ranked = self.rank_reaction(
                    rid,
                    orientation=orientation,
                    user_text="整条路径兼容性评估：为这一步保留模型优先候选，供全局兼容性重排。",
                    route_mode="default",
                    top_k=10,
                    confirmed_seed_ids=[],
                )
                rows = [dict(row) for row in (ranked.get("candidates") or [])[:5]]
                ranking_meta = {
                    "source": "catalyst_finder_r2e",
                    "route_id": (ranked.get("ranking") or {}).get("route_id"),
                    "candidate_count": len(rows),
                }
            if not rows:
                raise ValueError(f"no enzyme candidates available for step {index}")
            for row in rows:
                row["local_rank"] = int(row.get("rank") or 1)
                row["step_index"] = index
            step_candidates.append(rows)
            resolved_steps.append({
                "step_index": index,
                "rhea_id": rid,
                "orientation": orientation,
                "equation": str(step.get("equation") or "").strip() or None,
                "explicit_enzyme_id": explicit_enzyme or None,
                "ranking": ranking_meta,
            })

        self._enrich_candidates(step_candidates)

        # Beam search over combinations. Compatibility evidence is allowed to
        # reorder close model candidates, but model utility remains the dominant term.
        beams: list[tuple[list[dict[str, Any]], float]] = [([], 0.0)]
        for rows in step_candidates:
            expanded: list[tuple[list[dict[str, Any]], float]] = []
            for selected, score in beams:
                for candidate in rows:
                    utility = _model_utility(candidate)
                    evidence_bonus = min(0.05, 0.008 * _coverage(candidate.get("condition_profile") or {}))
                    compatibility = 0.0
                    for previous in selected:
                        delta, _ = pairwise_compatibility(previous, candidate, mode=mode)
                        compatibility += delta
                    target_delta, _ = target_condition_compatibility(candidate, target)
                    expanded.append((selected + [candidate], score + 0.78 * utility + evidence_bonus + compatibility + target_delta))
            expanded.sort(key=lambda item: item[1], reverse=True)
            beams = expanded[:120]
        selected, global_score = beams[0]

        issues: list[dict[str, Any]] = []
        pathway_reactions = [str(step.get("rhea_id") or "") for step in resolved_steps]
        for i, left in enumerate(selected):
            _, target_issues = target_condition_compatibility(left, target)
            for issue in target_issues:
                issues.append({**issue, "steps": [i + 1], "enzymes": [left.get("candidate_id")]})

            cid = str(left.get("candidate_id") or "")
            known_pairs = self.catalog.pairs_by_protein.get(cid, []) if hasattr(self.catalog, "pairs_by_protein") else []
            known_other_steps = sorted({
                pathway_reactions[j]
                for j in range(len(pathway_reactions))
                if j != i and any(str(pair.get("reaction_id") or "") == pathway_reactions[j] for pair in known_pairs)
            })
            if known_other_steps:
                issues.append({
                    "severity": "info",
                    "type": "cross_step_activity",
                    "detail": f"当前知识库还把这个酶与路径中的其他步骤关联：{', '.join(known_other_steps)}。这可能是有用的多功能性，也可能带来串扰，建议结合底物特异性实验复核。",
                    "steps": [i + 1],
                    "enzymes": [cid],
                })

            for j in range(i + 1, len(selected)):
                _, pair_issues = pairwise_compatibility(left, selected[j], mode=mode)
                for issue in pair_issues:
                    issues.append({**issue, "steps": [i + 1, j + 1], "enzymes": [left.get("candidate_id"), selected[j].get("candidate_id")]})
                if cid and cid == str(selected[j].get("candidate_id") or ""):
                    issues.append({
                        "severity": "info",
                        "type": "shared_enzyme",
                        "detail": "同一个酶被多个步骤选中；这可能简化表达或加酶方案，但需要确认它在完整底物/中间体混合物中的选择性。",
                        "steps": [i + 1, j + 1],
                        "enzymes": [cid],
                    })

        # Shared condition windows are reported only across enzymes that actually
        # have evidence. Coverage is surfaced separately to avoid treating unknown
        # annotations as proof of compatibility.
        ph_intervals = [(c.get("condition_profile") or {}).get("ph_active") for c in selected]
        ph_known = [x for x in ph_intervals if x]
        temp_intervals = [(c.get("condition_profile") or {}).get("temperature_active_c") for c in selected]
        temp_known = [x for x in temp_intervals if x]
        # Never call a single-enzyme interval a shared pathway window. A shared
        # condition is reported only when every selected enzyme has that evidence.
        common_ph = _intersection(ph_known) if len(ph_known) == len(selected) and selected else None
        common_temp = _intersection(temp_known) if len(temp_known) == len(selected) and selected else None
        cofactor_sets = [
            {str(value) for value in (candidate.get("condition_profile") or {}).get("cofactors") or [] if str(value).strip()}
            for candidate in selected
        ]
        cofactor_known = [values for values in cofactor_sets if values]
        common_cofactors = sorted(set.intersection(*cofactor_known)) if len(cofactor_known) == len(selected) and selected else []

        selected_steps: list[dict[str, Any]] = []
        evidence_count = 0
        core_condition_count = 0
        for index, candidate in enumerate(selected):
            profile = candidate.get("condition_profile") or {}
            if profile.get("available") and _coverage(profile):
                evidence_count += 1
            if profile.get("ph_optimum") or profile.get("ph_active") or profile.get("temperature_optimum_c") or profile.get("temperature_active_c"):
                core_condition_count += 1
            local_best = step_candidates[index][0]
            selected_steps.append({
                **resolved_steps[index],
                "selected_enzyme": candidate,
                "local_best_id": local_best.get("candidate_id"),
                "changed_for_pathway_compatibility": candidate.get("candidate_id") != local_best.get("candidate_id"),
                "alternatives": [
                    {
                        "candidate_id": row.get("candidate_id"),
                        "name": row.get("name"),
                        "species": row.get("species"),
                        "local_rank": row.get("local_rank"),
                        "score": row.get("score"),
                        "condition_profile": row.get("condition_profile"),
                    }
                    for row in step_candidates[index][:3]
                    if row.get("candidate_id") != candidate.get("candidate_id")
                ],
            })

        high = sum(1 for row in issues if row.get("severity") == "high")
        medium = sum(1 for row in issues if row.get("severity") == "medium")
        coverage_fraction = evidence_count / len(selected) if selected else 0.0
        core_fraction = core_condition_count / len(selected) if selected else 0.0
        if high:
            verdict = "sequential_recommended"
            verdict_label = "存在明显条件冲突，建议分步或分区"
        elif medium >= 2:
            verdict = "needs_optimization"
            verdict_label = "存在兼容性风险，需要优化共同条件"
        elif mode == "sequential":
            verdict = "sequential_compatible"
            verdict_label = "分步执行可分别优化各步条件"
        elif core_fraction < 0.5:
            verdict = "insufficient_evidence"
            verdict_label = "pH / 温度条件证据不足"
        elif core_fraction < 1.0:
            verdict = "partial_evidence"
            verdict_label = "条件证据不完整，尚不能确认共存条件"
        else:
            verdict = "compatible_with_caveats"
            verdict_label = "未发现强冲突，但仍需实验确认"

        recommendations: list[str] = []
        if high or medium:
            if mode == "in_vivo":
                recommendations.append("优先替换冲突步骤的酶，或通过亚细胞区室化/靶向把不兼容步骤分开。")
            elif mode == "sequential":
                recommendations.append("保持分步执行，并在步骤之间按各酶证据调整 buffer、pH、温度或辅因子；必要时在切换前进行中间体处理。")
            else:
                recommendations.append("若共同 pH/温度窗口不足，优先采用 sequential cascade，在步骤间调整 buffer、pH 或温度。")
        if common_ph not in (None, []):
            recommendations.append(f"已报道条件的共同 pH 窗口约为 {_format_interval(common_ph)}；可作为小规模条件扫描中心，而不是直接当作最终工艺条件。")
        if common_temp not in (None, []):
            recommendations.append(f"已报道条件的共同温度窗口约为 {_format_interval(common_temp, suffix=' °C')}；建议围绕该区间做活性/稳定性矩阵。")
        if common_cofactors:
            recommendations.append(f"所有选中酶的 UniProt 注释都包含共同辅因子：{' / '.join(common_cofactors)}。这可作为起始条件线索，但浓度和与其他金属/辅因子的组合仍需优化。")
        missing_core = len(selected) - core_condition_count
        if missing_core > 0:
            recommendations.append(f"有 {missing_core} 个步骤缺少可直接比较的 pH / 温度注释。建议优先回查 BRENDA、SABIO-RK 或原始文献，并用小规模 pH × 温度活性/稳定性矩阵补齐证据。")
        changed = [row for row in selected_steps if row["changed_for_pathway_compatibility"]]
        if changed:
            recommendations.append(f"全局兼容性重排替换了 {len(changed)} 个步骤的局部 Top-1；这些替换应作为实验优先对照，而不是自动视为更优。")
        recommendations.append("“未发现冲突”不等于不会沉淀：真实沉淀/失活还受蛋白浓度、pI、盐、buffer、底物/产物、溶剂和时间影响，需要做混合稳定性实验。")

        route_nodes = [
            {"id": "pathway-parse", "title": "解析整条反应路径", "subtitle": "natural language → verified steps", "kind": "input", "metric": f"{len(steps)} steps", "detail": "把自然语言中的多步反应拆成可核对的 Rhea 步骤；不允许语言模型发明数据库 ID。"},
            {"id": "pathway-r2e", "title": "逐步生成候选酶", "subtitle": "reuse Catalyst Finder R2E", "kind": "model", "metric": f"top ≤5 × {len(steps)}", "detail": "复用现有反应→酶生产排序，为每一步保留一小组局部优先候选。"},
            {"id": "pathway-uniprot-conditions", "title": "汇集酶条件证据", "subtitle": "UniProtKB curated annotations", "kind": "trust", "metric": f"annotations {evidence_count}/{len(steps)} · pH/T {core_condition_count}/{len(steps)}", "detail": "读取 UniProtKB 中的 pH、温度、辅因子、活性调控和亚细胞定位注释。缺失数据记为未知，不记为兼容。"},
            {"id": "pathway-global-rerank", "title": "全局兼容性重排", "subtitle": "beam search over enzyme sets", "kind": "fusion", "metric": f"score {global_score:.3f}", "detail": "在保持单步模型排序为主要信号的前提下，比较多步候选组合的条件兼容性。"},
            {"id": "pathway-conflict-audit", "title": "路径冲突审计", "subtitle": "pH · temperature · cofactor · localization", "kind": "filter", "metric": f"{high} high · {medium} medium", "detail": "显式列出支持或反对 one-pot / in-vivo 共存的证据，并给出分步、替换或区室化建议。"},
            {"id": "pathway-output", "title": "输出整条路径建议", "subtitle": "experiment-facing plan", "kind": "output", "metric": verdict_label, "detail": "输出逐步酶选择、共同条件窗口、冲突点和需要实验确认的未知项。"},
        ]
        return {
            "direction": "pathway_compatibility",
            "execution_mode": mode,
            "host": host or None,
            "target_conditions": target,
            "verdict": verdict,
            "verdict_label": verdict_label,
            "global_score": round(global_score, 4),
            "coverage": {
                "annotation_steps": evidence_count,
                "core_condition_steps": core_condition_count,
                "total_steps": len(selected),
                "annotation_fraction": round(coverage_fraction, 4),
                "core_condition_fraction": round(core_fraction, 4),
            },
            "shared_conditions": {
                "ph": common_ph,
                "ph_label": _format_interval(common_ph),
                "ph_coverage": len(ph_known),
                "temperature_c": common_temp,
                "temperature_label": _format_interval(common_temp, suffix=" °C"),
                "temperature_coverage": len(temp_known),
                "cofactors": common_cofactors,
                "cofactor_coverage": len(cofactor_known),
            },
            "steps": selected_steps,
            "conflicts": issues,
            "recommendations": recommendations,
            "evidence_sources": [
                {"name": "Catalyst Finder R2E", "role": "single-step enzyme candidate ranking"},
                {"name": "UniProtKB", "role": "curated enzyme condition/cofactor/localization evidence"},
            ],
            "limitations": [
                "当前没有把缺失的 pH/温度注释当作兼容证据。",
                "当前不直接预测蛋白沉淀、聚集或长期失活；这些需要与浓度、buffer、盐、pI、底物/产物和时间共同实验验证。",
                "SABIO-RK/BRENDA 可作为额外条件证据源，但当前部署未把不可用或需要凭据的服务硬绑定到主流程。",
                "整条路径的热力学/FBA 评价属于下一层 pathway feasibility；本模块聚焦已给定路径上的酶组合兼容性。",
            ],
            "route_view": {
                "direction": "pathway_compatibility",
                "route_id": "pathway-compatibility-v1",
                "base_route_id": "pathway-compatibility-v1",
                "active_overlays": [],
                "title": "整条路径 · 多酶兼容性评估",
                "summary": "逐步复用生产 R2E 排序，再用 UniProt 条件证据对整条路径的酶组合做全局兼容性重排。",
                "nodes": route_nodes,
                "edges": [{"from": route_nodes[i]["id"], "to": route_nodes[i+1]["id"]} for i in range(len(route_nodes)-1)],
                "decision": {"mode": mode, "host": host or None, "target_conditions": target, "steps": len(steps)},
            },
        }
