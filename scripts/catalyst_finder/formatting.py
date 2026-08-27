from __future__ import annotations

import re
from typing import Any


def ui_language(value: Any) -> str:
    return "zh" if str(value or "").strip().lower().startswith("zh") else "en"


def lang_text(language: str, en: str, zh: str) -> str:
    return zh if ui_language(language) == "zh" else en


def probable_uniprot(identifier: str) -> str:
    value = str(identifier or "").strip().upper()
    if re.fullmatch(r"[A-Z0-9]{6}", value) and any(ch.isdigit() for ch in value):
        return value
    if re.fullmatch(r"[A-Z0-9]{10}", value) and any(ch.isdigit() for ch in value):
        return value
    return ""
