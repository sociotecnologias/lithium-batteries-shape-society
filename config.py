from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

DB_DIR = PROJECT_ROOT / "databases"
CSV_DIR = PROJECT_ROOT / "input"
OUT_DIR = PROJECT_ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"
TIDY_DIR = OUT_DIR / "tidy"

SCOPUS_CSV = CSV_DIR / "scopus_20_8_25.csv"
STOPWORDS_XLSX = DB_DIR / "Database_stopwords_en.xlsx"
SOURCES_XLSX = DB_DIR / "Database_sources_final.xlsx"
AFI_NA_XLSX = DB_DIR / "Database_afi_na.xlsx"
MACROAREAS_XLSX = DB_DIR / "Database_Macroareas.xlsx"

MACRO_PALETTE = {
    "Administration and Decision Sciences": "lightpink",
    "Engineering and Technology":            "lightblue",
    "Health and Medical Sciences":           "lightgreen",
    "Multidisciplinary":                     "grey",
    "Natural and Physical Sciences":         "purple",
    "Social Sciences and Humanities":        "brown",
}

ECON_LEVELS = [
    "1. Developed region: G7",
    "2. Developed region: nonG7",
    "3. Emerging region: BRIC",
    "4. Emerging region: MIKT",
    "5. Emerging region: G20",
    "6. Developing region",
    "7. Least developed region",
]

GRUPO3 = [
    "human health", "human", "social life", "social", "adoption", "tension",
    "human rights", "environmental policy", "governance", "climate policy",
    "regulation", "politic", "territory", "dispute", "socio-economic",
    "socioeconomic", "climate", "pollution", "renewable", "sustainability",
    "environmental", "e mobility", "mobility", "vehicle",
    "extractivism", "salar", "indigenous", "community", "just transition"
]

AGENDA_KEYWORDS = {
    "Geopolitics": ["geopolitic", "global relation", "international", "supply chain", "china", "united states", "interdependence", "inequality"],
    "Policies": ["policy", "plan", "incentive", "regulation", "standard", "governance", "institution", "promotion"],
    "Practices": ["practice", "valuation", "use", "adoption", "consumption", "recycling", "behavior", "user"],
    "Conflicts": ["conflict", "controversy", "acceptability", "socio-environmental", "impact", "tension", "indigenous", "territory"],
    "Imaginaries": ["imaginary", "future", "transition", "paradigm", "expectation"]
}


EXTRA_STOPWORDS = ["inf", "<inf>", "©", "h<inf>", "hinf"]
STOP_PHRASES = ["© elsevier b v all rights reserved", "< inf>"]

FIX_TO_WORLD = {
    "United States":    "United States of America",
    "Serbia":           "Republic of Serbia",
    "Baden":            "Germany",
    "Hamburg":          "Germany",
    "Brunswick":        "Germany",
    "Oldenburg":        "Germany",
    "Modena":           "Italy",
    "Parma":            "Italy",
    "Réunion":          "France",
    "Hong Kong S.A.R.": "Hong Kong",
}

AFFIL_REPLACEMENTS = {
    "Czech Republic":     "Czechia",
    "Viet Nam":           "Vietnam",
    "Russian Federation": "Russia",
    "USA":                "United States",
    "Türkiye":            "Turkey",
}
