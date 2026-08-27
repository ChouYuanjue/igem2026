from __future__ import annotations

import re

UNIPROT_MENTION_RE = re.compile(
    r"\bUniProt(?:KB)?\s*[:#]?\s*([A-Z0-9]{6}(?:[A-Z0-9]{4})?)\b",
    re.IGNORECASE,
)


def explicit_uniprot_accession(text: str) -> str:
    """Return an accession only when the user explicitly labels it as UniProt."""
    match = UNIPROT_MENTION_RE.search(str(text or ""))
    return match.group(1).upper() if match else ""


def unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        key = value.casefold().strip()
        if key and key not in seen:
            seen.add(key)
            result.append(value.strip())
    return result


def quote_rhea_term(value: str) -> str:
    value = value.strip()
    if not value:
        return ""
    if re.match(r"^(?:chebi|inchikey|cas|rhea-comp):", value, re.IGNORECASE):
        return value
    if any(ch.isspace() for ch in value):
        return f'"{value.replace(chr(34), "").strip()}"'
    return value


def fallback_queries(substrates: list[str], products: list[str]) -> list[str]:
    queries = []
    for substrate in substrates[:3]:
        for product in products[:3]:
            queries.append(f"{quote_rhea_term(substrate)} AND {quote_rhea_term(product)}")
    queries.extend(quote_rhea_term(value) for value in products[:3])
    queries.extend(quote_rhea_term(value) for value in substrates[:3])
    return [query for query in queries if query]


def norm_text(text: str) -> str:
    text = text.casefold().replace("β", "beta").replace("α", "alpha")
    text = re.sub(r"[\[\]{}()'\";,._:+\-/\\]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def term_score(term: str, side: str) -> float:
    term_n = norm_text(term)
    side_n = norm_text(side)
    if not term_n or not side_n:
        return 0.0
    if term_n in side_n:
        return 4.0 + min(1.2, len(term_n) / 24.0)
    stop = {"a", "an", "the", "of", "and", "ion", "acid", "compound"}
    term_tokens = {token for token in term_n.split() if len(token) > 1 and token not in stop}
    side_tokens = set(side_n.split())
    if not term_tokens:
        return 0.0
    overlap = len(term_tokens & side_tokens) / len(term_tokens)
    return overlap * 2.6


def side_score(terms: list[str], side: str) -> float:
    if not terms:
        return 0.0
    scores = sorted((term_score(term, side) for term in terms), reverse=True)
    return scores[0] + (scores[1] * 0.25 if len(scores) > 1 else 0.0)


def candidate_match(
    equation: str,
    substrates: list[str],
    products: list[str],
) -> tuple[float, str]:
    parts = re.split(r"\s+(?:<=>|=>|<=|=|→|↔)\s+", equation, maxsplit=1)
    if len(parts) != 2:
        return side_score(substrates + products, equation), "forward"
    left, right = parts
    forward = side_score(substrates, left) + side_score(products, right)
    reverse = side_score(substrates, right) + side_score(products, left)
    if reverse > forward:
        return reverse, "reverse"
    return forward, "forward"
