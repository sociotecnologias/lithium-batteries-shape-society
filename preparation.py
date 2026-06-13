"""
Python port of direct_preparation.R

Loads the raw Scopus CSV, cleans Abstract / Author Keywords / Title columns,
filters by topical keywords (grupo3), merges sources & macroareas, extracts
country names from Affiliations, and produces both wide (procdata_final) and
long (procdata_long_final) tables.
"""
from __future__ import annotations

import re
import string
from typing import Iterable

import pandas as pd
import pycountry

from config import (
    SCOPUS_CSV, STOPWORDS_XLSX, SOURCES_XLSX, AFI_NA_XLSX, MACROAREAS_XLSX,
    GRUPO3, EXTRA_STOPWORDS, STOP_PHRASES, AFFIL_REPLACEMENTS, TIDY_DIR,
)


# ---------------------------------------------------------------------------
# Text cleaning helpers
# ---------------------------------------------------------------------------

_PUNCT_RE = re.compile(r"[" + re.escape(string.punctuation) + r"]")
_DIGIT_RE = re.compile(r"\d")
_SPACE_RE = re.compile(r"\s+")
_HTTP_RE = re.compile(r"http\S*")
_HASH_RE = re.compile(r"#\S*")
_AT_RE = re.compile(r"@\S*")


def limpiar(texto) -> str:
    """Mirrors the R `limpiar` function: lowercase, strip urls/handles/hashtags,
    punctuation, digits, and collapse whitespace."""
    if pd.isna(texto):
        return ""
    s = str(texto).lower()
    s = _HTTP_RE.sub("", s)
    s = _HASH_RE.sub("", s)
    s = _AT_RE.sub("", s)
    s = _PUNCT_RE.sub(" ", s)
    s = _DIGIT_RE.sub(" ", s)
    s = _SPACE_RE.sub(" ", s)
    return s


def load_stopwords() -> list[str]:
    sw = pd.read_excel(STOPWORDS_XLSX, header=None)
    words = sw.iloc[1:, 0].dropna().astype(str).str.strip().str.lower().tolist()
    first = str(sw.iloc[0, 0]).strip().lower()
    if first and first != "nan":
        words.insert(0, first)
    words.append("inf")
    extra = [w.strip().lower() for w in EXTRA_STOPWORDS]
    out = []
    seen = set()
    for w in words + extra:
        if w and w not in seen:
            seen.add(w)
            out.append(w)
    return out


def _build_stopword_pattern(stop_words: Iterable[str]) -> re.Pattern:
    escaped = [re.escape(w) for w in stop_words if w]
    return re.compile(r"\b(" + "|".join(escaped) + r")\b")


def clean_text_column(series: pd.Series, stop_pattern: re.Pattern) -> pd.Series:
    cleaned = series.map(limpiar)
    for phrase in STOP_PHRASES:
        cleaned = cleaned.str.replace(phrase, "", regex=False)
    cleaned = cleaned.str.replace("<", "", regex=False)
    cleaned = cleaned.str.replace(">", "", regex=False)
    cleaned = cleaned.str.lower()
    cleaned = cleaned.map(lambda t: stop_pattern.sub("", t) if isinstance(t, str) else "")
    cleaned = cleaned.map(lambda t: _SPACE_RE.sub(" ", t).strip())
    return cleaned


# ---------------------------------------------------------------------------
# Country extraction
# ---------------------------------------------------------------------------

def _country_list() -> list[str]:
    """Approximation of `countrycode::codelist$country.name.en`."""
    names = sorted({c.name for c in pycountry.countries})
    extras = [
        "United States", "Russia", "South Korea", "North Korea", "Vietnam",
        "Czechia", "Turkey", "Iran", "Bolivia", "Venezuela", "Tanzania",
        "Moldova", "Syria", "Laos", "Brunei", "Cape Verde", "Ivory Coast",
        "Macedonia", "Micronesia", "Palestine", "Taiwan", "Hong Kong",
        "Macao", "United Kingdom",
    ]
    for e in extras:
        if e not in names:
            names.append(e)
    return names


_REGEX_META_RE = re.compile(r"([\^$.|?*+(){}\[\]\\])")


def _escape_regex(s: str) -> str:
    return _REGEX_META_RE.sub(r"\\\1", s)


def _build_country_patterns(countries: list[str]) -> dict[str, re.Pattern]:
    return {
        c: re.compile(rf"(?<![A-Za-z]){_escape_regex(c)}(?![A-Za-z])", re.IGNORECASE)
        for c in countries
    }


def extract_countries(texto, patterns: dict[str, re.Pattern]) -> str | float:
    if not isinstance(texto, str) or not texto:
        return pd.NA
    positions: dict[str, int] = {}
    for name, pat in patterns.items():
        m = pat.search(texto)
        if m is not None:
            positions[name] = m.start()
    if not positions:
        return pd.NA
    ordered = sorted(positions.items(), key=lambda kv: kv[1])
    seen = set()
    unique_ordered: list[str] = []
    for name, _ in ordered:
        if name not in seen:
            seen.add(name)
            unique_ordered.append(name)
    return "; ".join(unique_ordered)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def prepare(write_outputs: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    # ---- DATA LOAD ----
    data = pd.read_csv(SCOPUS_CSV, sep=",")
    rename_map = {
        "Source title": "Source_Title",
        "Cited by": "Cited_by",
        "Author Keywords": "Autor_KW",
    }
    data = data.rename(columns=rename_map)
    keep = ["Authors", "Title", "Year", "Source_Title", "Cited_by", "DOI",
            "Affiliations", "Abstract", "Autor_KW", "Publisher", "ISSN"]
    procdata = data[keep].copy()

    # ---- STOPWORDS ----
    stop_words = load_stopwords()
    pattern = _build_stopword_pattern(stop_words)

    # ---- TEXT CLEANING ----
    procdata["clean_abs"] = clean_text_column(procdata["Abstract"], pattern)
    procdata["clean_kw"] = clean_text_column(procdata["Autor_KW"], pattern)
    procdata["clean_title"] = clean_text_column(procdata["Title"], pattern)

    # ---- CHECKPOINT 1: filter rows that mention any grupo3 term ----
    cols_for_check = ["clean_title", "clean_abs", "clean_kw"]
    grupo3_pattern = re.compile("|".join(re.escape(t) for t in GRUPO3))
    mask = procdata[cols_for_check].apply(
        lambda row: any(
            isinstance(v, str) and bool(grupo3_pattern.search(v)) for v in row
        ),
        axis=1,
    )
    procdata["Estado"] = mask.map(lambda b: "Sí" if b else "No")
    procdata = procdata[procdata["Estado"] == "Sí"].copy()

    # ---- SOURCES MERGE ----
    sources_final = pd.read_excel(SOURCES_XLSX)
    procdata = procdata.merge(sources_final, on="Source_Title", how="left")
    procdata = procdata[procdata["Area"].notna()].copy()
    procdata = procdata.reset_index(drop=True)
    procdata["ID"] = range(1, len(procdata) + 1)

    # ---- COUNTRY EXTRACTION ----
    afi_id = procdata[["Affiliations", "ID"]].copy()
    for old, new in AFFIL_REPLACEMENTS.items():
        afi_id["Affiliations"] = afi_id["Affiliations"].astype(str).str.replace(
            old, new, regex=False
        )
        procdata["Affiliations"] = procdata["Affiliations"].astype(str).str.replace(
            old, new, regex=False
        )

    paises = _country_list()
    paises = ["Hong Kong" if p == "Hong Kong SAR China" else p for p in paises]
    patterns = _build_country_patterns(paises)
    afi_id["Country"] = afi_id["Affiliations"].map(
        lambda t: extract_countries(t, patterns)
    )

    # ---- AFFILIATION OVERRIDES ----
    afi_extras = pd.read_excel(AFI_NA_XLSX)
    afi_extras = afi_extras.drop(columns=["Affiliations"], errors="ignore")
    merged = afi_id.merge(
        afi_extras[["ID", "Country"]], on="ID", how="left",
        suffixes=("_x", "_y"),
    )
    merged["Country_affiliation"] = merged["Country_x"].fillna(merged["Country_y"])
    afi_final = merged.drop(columns=["Country_x", "Country_y", "Affiliations"])

    procdata = procdata.merge(afi_final, on="ID", how="left")

    # ---- AREAS / MACROAREAS LONG TABLE ----
    macroareas = pd.read_excel(MACROAREAS_XLSX)
    expanded = procdata.assign(
        **{"Area.1": procdata["Area"].astype(str).str.split(";")}
    ).explode("Area.1")
    expanded["Area.1"] = expanded["Area.1"].astype(str).str.strip()
    procdata_long = expanded.merge(macroareas, on="Area.1", how="left")
    procdata_long = procdata_long.reset_index(drop=True)

    if write_outputs:
        TIDY_DIR.mkdir(parents=True, exist_ok=True)
        procdata.to_excel(TIDY_DIR / "procdata_final.xlsx", index=False)
        procdata_long.to_excel(TIDY_DIR / "procdata_long_final.xlsx", index=False)

    return procdata, procdata_long


if __name__ == "__main__":
    pd_wide, pd_long = prepare(write_outputs=True)
    print(f"procdata_final: {pd_wide.shape}")
    print(f"procdata_long_final: {pd_long.shape}")
