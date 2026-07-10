from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from rdkit import Chem


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_DATA_DIR = PROJECT_ROOT / "data" / "terpene"
TERPENE_DATA_DIR = PROJECT_ROOT / "data" / "terpene_cage_screen"
TERPENE_RESULTS_DIR = PROJECT_ROOT / "results" / "terpene_cage_screen"

SOURCE_FILES = {
    "positive_labels": SOURCE_DATA_DIR / "enzyme_terpene_synthase.tsv",
    "candidate_enzymes": SOURCE_DATA_DIR / "all_seq_terpene_synthase.tsv",
    "selected_reactions": SOURCE_DATA_DIR / "10rhea_selected.tsv",
}

UNIPROT_ACCESSION_RE = re.compile(
    r"^(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9][A-Z0-9]{3}[0-9]|A0A[A-Z0-9]{7})(?:-\d+)?$"
)
UNIPROT_ACCESSION_SEARCH_RE = re.compile(
    r"(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9][A-Z0-9]{3}[0-9]|A0A[A-Z0-9]{7})(?:-\d+)?"
)

ROLE_CANDIDATES = {
    "enzyme_id": [
        "enzyme_id",
        "protein_id",
        "uniprot_id",
        "uniprot",
        "entry",
        "accession",
        "uid",
        "id",
    ],
    "uniprot_id": [
        "uniprot_id",
        "uniprot",
        "entry",
        "accession",
        "protein_id",
        "uid",
    ],
    "sequence": [
        "sequence",
        "seq",
        "protein_sequence",
        "aa_sequence",
        "protein_seq",
    ],
    "rhea_id": [
        "rhea_id",
        "rhea",
        "rheaid",
        "rhea_identifier",
    ],
    "reaction_smiles": [
        "reaction_smiles",
        "cano_rxn_smiles",
        "smiles_seq",
        "smiles",
        "rxn_smiles",
        "reaction",
    ],
}


def ensure_parent_dir(path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    # Prefer the file extension, then fall back to sniffer-based detection.
    suffix = path.suffix.lower()
    if suffix in {".tsv", ".tab"}:
        sep = "\t"
    elif suffix == ".csv":
        sep = ","
    else:
        sample = path.read_text(encoding="utf-8", errors="ignore")[:8192]
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters="\t,;")
            sep = dialect.delimiter
        except Exception:
            sep = "\t"

    df = pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False)
    df.columns = [str(column).strip() for column in df.columns]
    return df


def write_table(df: pd.DataFrame, path: str | Path, sep: str = "\t") -> Path:
    path = ensure_parent_dir(Path(path))
    df.to_csv(path, sep=sep, index=False)
    return path


def first_rows(df: pd.DataFrame, n: int = 5) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, row in df.head(n).iterrows():
        item: dict[str, Any] = {}
        for key, value in row.to_dict().items():
            if pd.isna(value):
                item[str(key)] = None
            else:
                item[str(key)] = value
        rows.append(item)
    return rows


def coerce_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null"} else text


def _sample_nonempty_values(series: pd.Series, limit: int = 200) -> list[str]:
    values = [coerce_text(value) for value in series.tolist()]
    values = [value for value in values if value]
    if len(values) > limit:
        return values[:limit]
    return values


def is_uniprot_like(value: Any) -> bool:
    text = coerce_text(value)
    if not text:
        return False
    if "|" in text:
        parts = [part.strip() for part in text.split("|") if part.strip()]
        if any(UNIPROT_ACCESSION_RE.fullmatch(part.split("-")[0]) for part in parts):
            return True
    if UNIPROT_ACCESSION_RE.fullmatch(text):
        return True
    return bool(UNIPROT_ACCESSION_SEARCH_RE.search(text))


def parse_uniprot_id(value: Any) -> str | None:
    text = coerce_text(value)
    if not text:
        return None

    candidates: list[str] = [part.strip() for part in re.split(r"[|\s;,:]+", text) if part.strip()]
    if not candidates:
        candidates = [text]

    for candidate in candidates:
        cleaned = candidate.split(":", 1)[-1].strip()
        if cleaned.startswith(("sp_", "tr_")):
            cleaned = cleaned.split("_", 1)[-1]
        cleaned = cleaned.split("/")[0].strip()
        base = cleaned.split("-")[0].strip()
        if UNIPROT_ACCESSION_RE.fullmatch(cleaned):
            return base
        if UNIPROT_ACCESSION_RE.fullmatch(base):
            return base

    match = UNIPROT_ACCESSION_SEARCH_RE.search(text)
    if match:
        return match.group(0).split("-")[0]
    return None


def is_sequence_like(value: Any) -> bool:
    text = coerce_text(value).upper()
    if len(text) < 20:
        return False
    if any(ch.isspace() for ch in text):
        return False
    letters = re.sub(r"[^A-Z]", "", text)
    if len(letters) < 20:
        return False
    return len(letters) / max(len(text), 1) >= 0.95 and bool(re.fullmatch(r"[ACDEFGHIKLMNPQRSTVWYBXZJUO]+", letters))


def is_rhea_like(value: Any) -> bool:
    text = coerce_text(value).upper()
    return text.startswith("RHEA:")


def is_reaction_smiles_like(value: Any) -> bool:
    text = coerce_text(value)
    if not text:
        return False
    return ">>" in text and any(token in text for token in ("C", "O", "N", "=", "[", "]", "("))


def _score_column(series: pd.Series, predicate) -> float:
    values = _sample_nonempty_values(series)
    if not values:
        return 0.0
    matches = sum(1 for value in values if predicate(value))
    return matches / len(values)


def find_column(
    df: pd.DataFrame,
    role: str,
    *,
    min_score: float = 0.6,
) -> tuple[str | None, str]:
    candidates = ROLE_CANDIDATES.get(role, [])
    normalized = {normalize_name(column): column for column in df.columns}

    for candidate in candidates:
        key = normalize_name(candidate)
        if key in normalized:
            return normalized[key], "name"

    predicate = {
        "enzyme_id": is_uniprot_like,
        "uniprot_id": is_uniprot_like,
        "sequence": is_sequence_like,
        "rhea_id": is_rhea_like,
        "reaction_smiles": is_reaction_smiles_like,
    }.get(role)

    if predicate is None:
        return None, "missing"

    best_column: str | None = None
    best_score = 0.0
    for column in df.columns:
        score = _score_column(df[column], predicate)
        if score > best_score:
            best_score = score
            best_column = column

    if best_column is not None and best_score >= min_score:
        return best_column, f"value:{best_score:.2f}"
    return None, "missing"


def identify_terpene_columns(df: pd.DataFrame) -> dict[str, dict[str, str | None]]:
    found: dict[str, dict[str, str | None]] = {}
    for role in ["enzyme_id", "uniprot_id", "sequence", "rhea_id", "reaction_smiles"]:
        column, source = find_column(df, role)
        found[role] = {"column": column, "source": source}

    if found["enzyme_id"]["column"] is None and found["uniprot_id"]["column"] is not None:
        found["enzyme_id"] = {
            "column": found["uniprot_id"]["column"],
            "source": "fallback:uniprot_id",
        }
    if found["uniprot_id"]["column"] is None and found["enzyme_id"]["column"] is not None:
        found["uniprot_id"] = {
            "column": found["enzyme_id"]["column"],
            "source": "fallback:enzyme_id",
        }
    return found


def canonicalize_reaction_smiles(reaction_smiles: str | None, remove_stereo: bool = True) -> str | None:
    text = coerce_text(reaction_smiles)
    if not text:
        return None
    if ">>" not in text:
        return text

    parts = text.split(">")
    if len(parts) < 3:
        return text

    reactant_text = parts[0]
    product_text = parts[-1]
    reactants = [part.strip() for part in reactant_text.split(".") if part.strip()]
    products = [part.strip() for part in product_text.split(".") if part.strip()]
    if not reactants or not products:
        return text

    canonical_parts: list[str] = []
    for side in (reactants, products):
        converted: list[str] = []
        for smiles in side:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return text
            if remove_stereo:
                Chem.RemoveStereochemistry(mol)
            converted.append(Chem.MolToSmiles(mol))
        canonical_parts.append(".".join(sorted(converted)))

    return ">>".join(canonical_parts)


def dedupe_preserve_order(values: Iterable[Any]) -> list[Any]:
    seen: set[Any] = set()
    ordered: list[Any] = []
    for value in values:
        if value in seen:
            continue
        ordered.append(value)
        seen.add(value)
    return ordered


def safe_json_dump(payload: Any, path: str | Path) -> Path:
    path = ensure_parent_dir(Path(path))
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_existing_table(path: str | Path, sep: str = "\t") -> pd.DataFrame | None:
    path = Path(path)
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False)
    except Exception:
        return None


def write_markdown(path: str | Path, lines: list[str]) -> Path:
    path = ensure_parent_dir(Path(path))
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path


def count_nonempty(df: pd.DataFrame, column: str | None) -> int:
    if not column or column not in df.columns:
        return 0
    series = df[column].astype(str).map(coerce_text)
    return int((series != "").sum())


def resolve_java_home() -> Path | None:
    """Return a Java 17+ home if one is available."""
    candidates: list[Path] = []
    env_java_home = os.environ.get("JAVA_HOME")
    if env_java_home:
        candidates.append(Path(env_java_home))

    candidates.extend(
        Path(path)
        for path in [
            "/usr/lib/jvm/java-21-openjdk-amd64",
            "/usr/lib/jvm/java-17-openjdk-amd64",
            "/usr/lib/jvm/java-21-openjdk-arm64",
            "/usr/lib/jvm/java-17-openjdk-arm64",
        ]
    )

    java_bin = shutil.which("java")
    if java_bin:
        candidates.append(Path(java_bin).resolve().parent.parent)

    for candidate in candidates:
        java_bin_path = candidate / "bin" / "java"
        if not java_bin_path.exists():
            continue
        try:
            completed = subprocess.run(
                [str(java_bin_path), "-version"],
                capture_output=True,
                text=True,
                check=False,
            )
        except Exception:
            continue
        output = (completed.stderr or "") + (completed.stdout or "")
        match = re.search(r'version "(\d+)', output)
        if not match:
            continue
        if int(match.group(1)) >= 17:
            return candidate
    return None


def find_p2rank_wrapper() -> Path:
    candidates = [
        PROJECT_ROOT / "external_repos" / "p2rank" / "distro" / "prank",
        PROJECT_ROOT / "external_repos" / "p2rank" / "prank.sh",
        PROJECT_ROOT / "data" / "assets" / "p2rank" / "p2rank_2.5.1" / "prank",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not find a P2Rank wrapper script.")

