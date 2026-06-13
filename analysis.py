"""
Python port of direct_analysis.R

Generates Figures 1-6 from the prepared procdata / procdata_long tables and
saves them as PNG files into outputs/figures/.

Figures
-------
1. Article counts per Macro_area / year (B/W and color, with/without minor grid).
2. Sum of citations per Macro_area / year (B/W and color, with/without minor grid).
3. Country choropleths (weighted by 1/k and first-listed) + bars by economy.
4. Top-10 token frequencies per Macro_area.
5. Bigram networks + bar chart (B/W and color).
6. TF-IDF top-10 concepts per Macro_area.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import plotly.graph_objects as go
import plotly.express as px

from config import MACRO_PALETTE, ECON_LEVELS, FIG_DIR, FIX_TO_WORLD, AGENDA_KEYWORDS


_TOKEN_RE = re.compile(r"[A-Za-zÀ-ÿ]+")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save(fig, name: str):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    if hasattr(fig, "write_image"):
        fig.write_image(path, width=1100, height=700, scale=2)
    else:
        fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)


def _five_year_breaks(years: pd.Series) -> list[int]:
    y_min = int(years.min())
    y_max = int(years.max())
    first = math.ceil(y_min / 5) * 5
    return [y_min] + list(range(first, y_max + 1, 5))


def _tokenize(text) -> list[str]:
    if not isinstance(text, str):
        return []
    return _TOKEN_RE.findall(text.lower())


# ---------------------------------------------------------------------------
# Figure 1 — Article counts over time
# ---------------------------------------------------------------------------

def _line_plot(df, y_col, title, palette_dict, breaks, fname,
               bw_styles=None, minor_grid=False):
    fig, ax = plt.subplots(figsize=(10, 6))
    macros = sorted(df["Macro_areas"].dropna().unique())
    for i, m in enumerate(macros):
        sub = df[df["Macro_areas"] == m].sort_values("Year")
        if bw_styles is not None and i < len(bw_styles):
            color, ls = bw_styles[i]
        else:
            color = palette_dict.get(m, "black")
            ls = "solid"
        ax.plot(sub["Year"], sub[y_col], label=m, color=color, linestyle=ls)
    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.set_ylabel(y_col.replace("_", " "))
    ax.set_xticks(breaks)
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    if minor_grid:
        ax.set_xticks(range(int(df["Year"].min()), int(df["Year"].max()) + 1),
                      minor=True)
        ax.grid(which="major", axis="x", color="grey", alpha=0.5)
        ax.grid(which="minor", axis="x", color="grey", alpha=0.2)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3,
              frameon=False, fontsize=8)
    fig.tight_layout()
    _save(fig, fname)


def figure_1(procdata_long: pd.DataFrame):
    counts = (
        procdata_long.groupby(["Year", "Macro_areas"], dropna=True)
        .size()
        .reset_index(name="Count")
        .dropna()
    )
    counts2 = counts[~counts["Macro_areas"].isin(
        ["Engineering and Technology", "Natural and Physical Sciences"]
    )]

    breaks = _five_year_breaks(counts["Year"])
    breaks2 = _five_year_breaks(counts2["Year"]) if not counts2.empty else breaks

    _line_plot(counts, "Count", "Macro areas frequency over time",
               MACRO_PALETTE, breaks, "fig1_A1_1.png")
    _line_plot(counts2, "Count", "Macro areas frequency over time",
               MACRO_PALETTE, breaks2, "fig1_A1_2.png")
    _line_plot(counts, "Count", "Macro areas frequency over time",
               MACRO_PALETTE, breaks, "fig1_B1_1.png", minor_grid=True)
    _line_plot(counts2, "Count", "Macro areas frequency over time",
               MACRO_PALETTE, breaks2, "fig1_B1_2.png", minor_grid=True)


# ---------------------------------------------------------------------------
# Figure 2 — Sum of citations
# ---------------------------------------------------------------------------

def figure_2(procdata_long: pd.DataFrame):
    sums = (
        procdata_long.groupby(["Year", "Macro_areas"], dropna=False)["Cited_by"]
        .sum()
        .reset_index(name="SumCitedBy")
        .dropna()
    )
    sums2 = sums[~sums["Macro_areas"].isin(
        ["Engineering and Technology", "Natural and Physical Sciences"]
    )]

    breaks = _five_year_breaks(sums["Year"])
    breaks2 = _five_year_breaks(sums2["Year"]) if not sums2.empty else breaks

    _line_plot(sums, "SumCitedBy", "Sum of citations by Macro Areas over time",
               MACRO_PALETTE, breaks, "fig2_A2_1.png")
    _line_plot(sums, "SumCitedBy", "Sum of citations by Macro Areas over time",
               MACRO_PALETTE, breaks, "fig2_B2_1.png", minor_grid=True)
    _line_plot(sums2, "SumCitedBy", "Sum of citations by Macro_areas over time",
               MACRO_PALETTE, breaks2, "fig2_A2_2.png")
    _line_plot(sums2, "SumCitedBy", "Sum of citations by Macro_areas over time",
               MACRO_PALETTE, breaks2, "fig2_B2_2.png", minor_grid=True)


# ---------------------------------------------------------------------------
# Figure 3 — Maps & economy bars (Plotly choropleth)
# ---------------------------------------------------------------------------

# Mapping from "geounit" naming used in the R script to ISO-3 codes that
# Plotly's choropleth (locationmode="ISO-3") accepts.
GEOUNIT_TO_ISO3 = {
    "United States of America": "USA", "United States": "USA",
    "Canada": "CAN", "Mexico": "MEX", "Brazil": "BRA", "Argentina": "ARG",
    "Chile": "CHL", "Peru": "PER", "Colombia": "COL", "Venezuela": "VEN",
    "Ecuador": "ECU", "Bolivia": "BOL", "Uruguay": "URY", "Paraguay": "PRY",
    "Cuba": "CUB", "Dominican Republic": "DOM", "Costa Rica": "CRI",
    "Panama": "PAN", "Guatemala": "GTM", "Honduras": "HND", "Nicaragua": "NIC",
    "El Salvador": "SLV", "Haiti": "HTI", "Jamaica": "JAM",
    "Trinidad and Tobago": "TTO", "Bahamas": "BHS",
    "United Kingdom": "GBR", "Ireland": "IRL", "France": "FRA", "Germany": "DEU",
    "Italy": "ITA", "Spain": "ESP", "Portugal": "PRT", "Netherlands": "NLD",
    "Belgium": "BEL", "Luxembourg": "LUX", "Switzerland": "CHE", "Austria": "AUT",
    "Denmark": "DNK", "Sweden": "SWE", "Norway": "NOR", "Finland": "FIN",
    "Iceland": "ISL", "Poland": "POL", "Czechia": "CZE", "Slovakia": "SVK",
    "Hungary": "HUN", "Romania": "ROU", "Bulgaria": "BGR", "Greece": "GRC",
    "Republic of Serbia": "SRB", "Serbia": "SRB", "Croatia": "HRV",
    "Slovenia": "SVN", "Bosnia and Herzegovina": "BIH", "Albania": "ALB",
    "Montenegro": "MNE", "North Macedonia": "MKD", "Macedonia": "MKD",
    "Estonia": "EST", "Latvia": "LVA", "Lithuania": "LTU",
    "Belarus": "BLR", "Ukraine": "UKR", "Moldova": "MDA",
    "Russia": "RUS", "Russian Federation": "RUS",
    "Turkey": "TUR", "Cyprus": "CYP", "Malta": "MLT",
    "China": "CHN", "Japan": "JPN", "South Korea": "KOR", "North Korea": "PRK",
    "Mongolia": "MNG", "Taiwan": "TWN", "Hong Kong": "HKG", "Macao": "MAC",
    "Vietnam": "VNM", "Thailand": "THA", "Cambodia": "KHM", "Laos": "LAO",
    "Myanmar": "MMR", "Malaysia": "MYS", "Singapore": "SGP", "Indonesia": "IDN",
    "Philippines": "PHL", "Brunei": "BRN", "Timor-Leste": "TLS",
    "India": "IND", "Pakistan": "PAK", "Bangladesh": "BGD", "Sri Lanka": "LKA",
    "Nepal": "NPL", "Bhutan": "BTN", "Afghanistan": "AFG", "Maldives": "MDV",
    "Iran": "IRN", "Iraq": "IRQ", "Syria": "SYR", "Jordan": "JOR",
    "Lebanon": "LBN", "Israel": "ISR", "Palestine": "PSE",
    "Saudi Arabia": "SAU", "Yemen": "YEM", "Oman": "OMN",
    "United Arab Emirates": "ARE", "Qatar": "QAT", "Bahrain": "BHR",
    "Kuwait": "KWT",
    "Kazakhstan": "KAZ", "Uzbekistan": "UZB", "Turkmenistan": "TKM",
    "Kyrgyzstan": "KGZ", "Tajikistan": "TJK", "Azerbaijan": "AZE",
    "Armenia": "ARM", "Georgia": "GEO",
    "Egypt": "EGY", "Libya": "LBY", "Tunisia": "TUN", "Algeria": "DZA",
    "Morocco": "MAR", "Sudan": "SDN", "South Sudan": "SSD",
    "Ethiopia": "ETH", "Eritrea": "ERI", "Somalia": "SOM", "Djibouti": "DJI",
    "Kenya": "KEN", "Tanzania": "TZA", "Uganda": "UGA", "Rwanda": "RWA",
    "Burundi": "BDI", "Nigeria": "NGA", "Niger": "NER", "Chad": "TCD",
    "Cameroon": "CMR", "Central African Republic": "CAF", "Gabon": "GAB",
    "Republic of the Congo": "COG", "Democratic Republic of the Congo": "COD",
    "Angola": "AGO", "Zambia": "ZMB", "Zimbabwe": "ZWE", "Malawi": "MWI",
    "Mozambique": "MOZ", "Namibia": "NAM", "Botswana": "BWA",
    "South Africa": "ZAF", "Lesotho": "LSO", "Eswatini": "SWZ",
    "Madagascar": "MDG", "Mauritius": "MUS", "Comoros": "COM",
    "Senegal": "SEN", "Gambia": "GMB", "Guinea-Bissau": "GNB", "Guinea": "GIN",
    "Sierra Leone": "SLE", "Liberia": "LBR", "Ivory Coast": "CIV",
    "Côte d'Ivoire": "CIV", "Ghana": "GHA", "Togo": "TGO", "Benin": "BEN",
    "Burkina Faso": "BFA", "Mali": "MLI", "Mauritania": "MRT",
    "Australia": "AUS", "New Zealand": "NZL", "Papua New Guinea": "PNG",
    "Fiji": "FJI", "Solomon Islands": "SLB", "Vanuatu": "VUT",
}

# Lightweight country -> economy classification used for the bar charts.
# This avoids depending on rnaturalearth's geounit/economy table.
ECONOMY_CLASS = {
    # 1. Developed region: G7
    "USA": "1. Developed region: G7", "GBR": "1. Developed region: G7",
    "FRA": "1. Developed region: G7", "DEU": "1. Developed region: G7",
    "ITA": "1. Developed region: G7", "JPN": "1. Developed region: G7",
    "CAN": "1. Developed region: G7",
    # 2. Developed region: nonG7
    "ESP": "2. Developed region: nonG7", "PRT": "2. Developed region: nonG7",
    "NLD": "2. Developed region: nonG7", "BEL": "2. Developed region: nonG7",
    "LUX": "2. Developed region: nonG7", "CHE": "2. Developed region: nonG7",
    "AUT": "2. Developed region: nonG7", "DNK": "2. Developed region: nonG7",
    "SWE": "2. Developed region: nonG7", "NOR": "2. Developed region: nonG7",
    "FIN": "2. Developed region: nonG7", "ISL": "2. Developed region: nonG7",
    "IRL": "2. Developed region: nonG7", "GRC": "2. Developed region: nonG7",
    "CYP": "2. Developed region: nonG7", "MLT": "2. Developed region: nonG7",
    "AUS": "2. Developed region: nonG7", "NZL": "2. Developed region: nonG7",
    "ISR": "2. Developed region: nonG7", "SGP": "2. Developed region: nonG7",
    "KOR": "2. Developed region: nonG7", "TWN": "2. Developed region: nonG7",
    "HKG": "2. Developed region: nonG7",
    # 3. Emerging region: BRIC
    "BRA": "3. Emerging region: BRIC", "RUS": "3. Emerging region: BRIC",
    "IND": "3. Emerging region: BRIC", "CHN": "3. Emerging region: BRIC",
    # 4. Emerging region: MIKT
    "MEX": "4. Emerging region: MIKT", "IDN": "4. Emerging region: MIKT",
    "TUR": "4. Emerging region: MIKT",
    # 5. Emerging region: G20
    "ARG": "5. Emerging region: G20", "ZAF": "5. Emerging region: G20",
    "SAU": "5. Emerging region: G20",
    # 6. Developing region (rest of mid-income)
    **{c: "6. Developing region" for c in [
        "CHL", "COL", "PER", "ECU", "URY", "PRY", "BOL", "VEN", "CUB", "DOM",
        "CRI", "PAN", "GTM", "HND", "NIC", "SLV", "JAM", "TTO", "BHS",
        "POL", "CZE", "SVK", "HUN", "ROU", "BGR", "SRB", "HRV", "SVN",
        "BIH", "ALB", "MNE", "MKD", "EST", "LVA", "LTU", "BLR", "UKR",
        "MDA", "GEO", "ARM", "AZE", "KAZ", "UZB", "TKM", "KGZ", "TJK",
        "EGY", "TUN", "MAR", "DZA", "JOR", "LBN", "IRN", "IRQ", "OMN",
        "ARE", "QAT", "BHR", "KWT", "PAK", "BGD", "LKA", "VNM", "THA",
        "MYS", "PHL", "MNG", "MUS", "GHA", "KEN", "NGA", "ZMB", "ZWE",
        "BWA", "NAM", "AGO", "MOZ", "NPL", "BTN", "MDV", "MMR",
    ]},
    # 7. Least developed region
    **{c: "7. Least developed region" for c in [
        "AFG", "YEM", "SYR", "PSE", "SDN", "SSD", "ETH", "ERI", "SOM",
        "DJI", "TZA", "UGA", "RWA", "BDI", "NER", "TCD", "CMR", "CAF",
        "GAB", "COG", "COD", "MWI", "MDG", "COM", "LSO", "SWZ", "SEN",
        "GMB", "GNB", "GIN", "SLE", "LBR", "CIV", "TGO", "BEN", "BFA",
        "MLI", "MRT", "PNG", "FJI", "SLB", "VUT", "TLS", "HTI", "KHM",
        "LAO", "BRN", "PRK",
    ]},
}


def _apply_fix(name):
    if not isinstance(name, str):
        return name
    return FIX_TO_WORLD.get(name, name)


def _explode_countries(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["_row_id"] = range(len(out))
    out["Country_affiliation"] = (
        out["Country_affiliation"].astype(str).str.split(r"\s*;\s*")
    )
    out = out.explode("Country_affiliation")
    out["Country_affiliation"] = (
        out["Country_affiliation"].str.replace(" ", " ", regex=False)
        .str.replace("​", " ", regex=False)
        .str.strip()
    )
    out = out[out["Country_affiliation"].notna() & (out["Country_affiliation"] != "")
              & (out["Country_affiliation"] != "nan")]
    out["Country_affiliation"] = out["Country_affiliation"].map(_apply_fix)
    return out


def figure_3(procdata: pd.DataFrame, procdata_long: pd.DataFrame):
    procdata = procdata.copy()
    procdata_long = procdata_long.copy()
    procdata["Country_affiliation"] = procdata["Country_affiliation"].map(_apply_fix)
    procdata_long["Country_affiliation"] = procdata_long["Country_affiliation"].map(_apply_fix)

    # ---------------- Weighted (1/k) choropleth ----------------
    proc_long_w = _explode_countries(procdata)
    k = proc_long_w.groupby("_row_id")["Country_affiliation"].transform("count")
    proc_long_w["w"] = 1.0 / k
    counts_weighted = (
        proc_long_w.groupby("Country_affiliation")["w"].sum().reset_index(name="n")
    )
    counts_weighted["iso3"] = counts_weighted["Country_affiliation"].map(GEOUNIT_TO_ISO3)
    counts_weighted = counts_weighted.dropna(subset=["iso3"])

    fig = px.choropleth(
        counts_weighted, locations="iso3", color="n",
        hover_name="Country_affiliation",
        color_continuous_scale=[(0, "#deebf7"), (1, "#08519c")],
        labels={"n": "Weight"},
        title="Weighted distribution by country (1/k per row)",
    )
    fig.update_layout(geo=dict(showframe=False))
    _save(fig, "fig3_A_weighted.png")

    # ---------------- First-listed country choropleth ----------------
    counts_first = procdata.copy()
    counts_first["first_country"] = (
        counts_first["Country_affiliation"].astype(str).str.split(";").str[0].str.strip()
    )
    counts_first = counts_first[counts_first["first_country"].notna()
                                & (counts_first["first_country"] != "")
                                & (counts_first["first_country"] != "nan")]
    counts_first = counts_first.groupby("first_country").size().reset_index(name="n")
    counts_first["iso3"] = counts_first["first_country"].map(GEOUNIT_TO_ISO3)
    counts_first = counts_first.dropna(subset=["iso3"])

    fig = px.choropleth(
        counts_first, locations="iso3", color="n",
        hover_name="first_country",
        color_continuous_scale=[(0, "#deebf7"), (1, "#08519c")],
        labels={"n": "Frequency"},
        title="Distribution by first listed country",
    )
    fig.update_layout(geo=dict(showframe=False))
    _save(fig, "fig3_B_first_color.png")

    # ---------------- Bars by economy + macro area (weighted) ----------------
    proc_long2 = _explode_countries(procdata_long.dropna(subset=["Macro_areas"]))
    k2 = proc_long2.groupby("_row_id")["Country_affiliation"].transform("count")
    proc_long2["w"] = 1.0 / k2
    proc_long2["iso3"] = proc_long2["Country_affiliation"].map(GEOUNIT_TO_ISO3)
    proc_long2["economy"] = proc_long2["iso3"].map(ECONOMY_CLASS)

    weighted_econ = (
        proc_long2.dropna(subset=["economy"])
        .groupby(["economy", "Macro_areas"])["w"].sum().reset_index(name="weight")
    )
    weighted_econ["economy"] = pd.Categorical(weighted_econ["economy"],
                                              categories=ECON_LEVELS, ordered=True)

    _bar_by_economy(weighted_econ, "weight", "Weighted count by economy (1/k per row)",
                    "Weighted count", "fig3_2_A_weighted.png", bw=False)

    # ---------------- Bars by economy + macro area (first-listed) ----------------
    procdata_long_clean = procdata_long.dropna(subset=["Macro_areas", "Country_affiliation"]).copy()
    procdata_long_clean["first_country"] = (
        procdata_long_clean["Country_affiliation"].astype(str).str.split(";").str[0].str.strip()
    )
    procdata_long_clean["first_country"] = procdata_long_clean["first_country"].map(_apply_fix)
    procdata_long_clean["iso3"] = procdata_long_clean["first_country"].map(GEOUNIT_TO_ISO3)
    procdata_long_clean["economy"] = procdata_long_clean["iso3"].map(ECONOMY_CLASS)

    first_econ = (
        procdata_long_clean.dropna(subset=["economy"])
        .groupby(["economy", "Macro_areas"]).size().reset_index(name="n")
    )
    first_econ["economy"] = pd.Categorical(first_econ["economy"],
                                           categories=ECON_LEVELS, ordered=True)

    _bar_by_economy(first_econ, "n", "First-listed country count by economy",
                    "Count", "fig3_2_B_first.png", bw=False)


def _bar_by_economy(df, value_col, title, ylabel, fname, bw):
    macros = sorted(df["Macro_areas"].dropna().unique())
    n = len(macros)
    if bw:
        greys = np.linspace(0.8, 0.2, max(n, 1))
        colors = [(g, g, g) for g in greys]
    else:
        colors = [MACRO_PALETTE.get(m, "grey") for m in macros]

    fig, ax = plt.subplots(figsize=(13, 7))
    econs = ECON_LEVELS
    x = np.arange(len(econs))
    width = 0.8 / max(n, 1)
    for i, m in enumerate(macros):
        sub = df[df["Macro_areas"] == m].set_index("economy")
        vals = [sub.loc[e, value_col] if e in sub.index else 0 for e in econs]
        ax.bar(x + i * width - 0.4 + width / 2, vals, width=width,
               label=m, color=colors[i])
    ax.set_xticks(x)
    ax.set_xticklabels(econs, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=3,
              frameon=False, fontsize=8)
    fig.subplots_adjust(bottom=0.35)
    _save(fig, fname)


# ---------------------------------------------------------------------------
# Figure 4 — Token frequency per macro area
# ---------------------------------------------------------------------------

_MACRO_TO_COLOR_FIG4 = {
    "Administration and Decision Sciences": "lightpink",
    "Engineering and Technology":            "lightblue",
    "Health and Medical Sciences":           "lightgreen",
    "Multidisciplinary":                     "grey",
    "Natural and Physical Sciences":         "purple",
    "Social Sciences and Humanities":        "brown",
}

_MACRO_TO_TITLE = {
    "Administration and Decision Sciences": "Decision Sciences",
    "Engineering and Technology":            "Engineering & Technology",
    "Health and Medical Sciences":           "Health Sciences",
    "Multidisciplinary":                     "Multidisciplinary",
    "Natural and Physical Sciences":         "Natural Sciences",
    "Social Sciences and Humanities":        "Social Sciences",
}


def _top_tokens(df: pd.DataFrame, text_col: str, n: int = 10) -> pd.DataFrame:
    counter: dict[str, int] = {}
    for txt in df[text_col].dropna():
        for tok in _tokenize(txt):
            counter[tok] = counter.get(tok, 0) + 1
    items = sorted(counter.items(), key=lambda kv: kv[1], reverse=True)[:n]
    return pd.DataFrame(items, columns=["word", "n"])


def figure_4(procdata_long: pd.DataFrame):
    for macro, color in _MACRO_TO_COLOR_FIG4.items():
        sub = procdata_long[procdata_long["Macro_areas"] == macro]
        top = _top_tokens(sub, "clean_abs", 10)
        if top.empty:
            continue
        title = _MACRO_TO_TITLE[macro]
        slug = title.replace(" ", "_").replace("&", "and")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(top["word"][::-1], top["n"][::-1], color=color)
        ax.set_xlabel("Frequency")
        ax.set_ylabel("Word")
        ax.set_title(title)
        fig.tight_layout()
        _save(fig, f"fig4_{slug}.png")


# ---------------------------------------------------------------------------
# Figure 5 — Bigrams (network + bar chart)
# ---------------------------------------------------------------------------

def _bigram_count(df: pd.DataFrame, col: str) -> pd.DataFrame:
    pairs: dict[tuple[str, str], int] = {}
    for txt in df[col].dropna():
        toks = _tokenize(txt)
        for w1, w2 in zip(toks, toks[1:]):
            key = (w1, w2)
            pairs[key] = pairs.get(key, 0) + 1
    rows = [(w1, w2, n) for (w1, w2), n in pairs.items()]
    out = pd.DataFrame(rows, columns=["word1", "word2", "weight"])
    out = out.sort_values("weight", ascending=False).reset_index(drop=True)
    return out


def _plot_bigram_network(bi: pd.DataFrame, title: str, ncon: int, fname: str,
                         vertex_color: str):
    sub = bi.iloc[:ncon, :3].copy()
    if sub.empty:
        return
    threshold = int(sub["weight"].iloc[-1]) - 1
    total = sub["weight"].sum()
    sub["percent"] = sub["weight"] / total
    sub = sub[sub["weight"] > threshold].copy()
    sub["scaled"] = sub["percent"] / 2e3

    G = nx.from_pandas_edgelist(sub, source="word1", target="word2",
                                edge_attr="scaled")
    if G.number_of_edges() == 0:
        return
    strength = dict(G.degree(weight="scaled"))
    max_w = max(d["scaled"] for _, _, d in G.edges(data=True))
    edge_widths = [10 * (d["scaled"] / max_w) for _, _, d in G.edges(data=True)]
    node_sizes = [max(50, 599999 * strength[n]) for n in G.nodes]

    fig, ax = plt.subplots(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42, k=0.8)
    nx.draw_networkx_edges(G, pos, ax=ax, width=edge_widths, edge_color="grey",
                           alpha=0.6)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=vertex_color,
                           node_size=node_sizes, edgecolors="black", linewidths=0.5)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=10)
    ax.set_title(f"{title}\nThreshold: {threshold}")
    ax.axis("off")
    fig.tight_layout()
    _save(fig, fname)


def figure_5(procdata: pd.DataFrame):
    bi_total = _bigram_count(procdata, "clean_abs")
    _plot_bigram_network(bi_total, "Total", 15, "fig5_1_network.png", "lightblue")

    top = bi_total.head(15).copy()
    top["bigram"] = top["word1"] + " " + top["word2"]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top["bigram"][::-1], top["weight"][::-1], color="lightblue")
    ax.set_xlim(0, 5300)
    ax.set_xticks(range(0, 5301, 1000))
    ax.set_xlabel("Weight")
    ax.set_ylabel("Bigram")
    ax.set_title("Top 15 most frequent bigrams")
    fig.tight_layout()
    _save(fig, "fig5_2_bars_color.png")


# ---------------------------------------------------------------------------
# Figure 6 — TF-IDF top concepts per macro area
# ---------------------------------------------------------------------------

def _tf_idf_table(procdata_long: pd.DataFrame) -> pd.DataFrame:
    """Replicates `bind_tf_idf(word, Macro_areas, n)` from R: per-document
    counts and total token counts grouped by Macro_areas, then standard
    tf*idf with idf = ln(N_docs / df_word)."""
    df = procdata_long.dropna(subset=["Macro_areas"]).copy()
    rows = []
    for macro, sub in df.groupby("Macro_areas"):
        counts: dict[str, int] = {}
        for txt in sub["clean_abs"].dropna():
            for tok in _tokenize(txt):
                counts[tok] = counts.get(tok, 0) + 1
        for w, n in counts.items():
            rows.append({"Macro_areas": macro, "word": w, "n": n})
    abs_words = pd.DataFrame(rows)
    if abs_words.empty:
        return abs_words
    totals = abs_words.groupby("Macro_areas")["n"].sum().reset_index(name="total")
    abs_words = abs_words.merge(totals, on="Macro_areas")
    abs_words["tf"] = abs_words["n"] / abs_words["total"]
    n_docs = abs_words["Macro_areas"].nunique()
    df_word = abs_words.groupby("word")["Macro_areas"].nunique().reset_index(
        name="df")
    abs_words = abs_words.merge(df_word, on="word")
    abs_words["idf"] = np.log(n_docs / abs_words["df"])
    abs_words["tf_idf"] = abs_words["tf"] * abs_words["idf"]
    return abs_words


def figure_6(procdata_long: pd.DataFrame):
    tf_idf = _tf_idf_table(procdata_long)
    if tf_idf.empty:
        return
    for macro, color in _MACRO_TO_COLOR_FIG4.items():
        sub = tf_idf[tf_idf["Macro_areas"] == macro]
        if sub.empty:
            continue
        top = sub.nlargest(10, "tf_idf").sort_values("tf_idf")
        title = _MACRO_TO_TITLE[macro]
        slug = title.replace(" ", "_").replace("&", "and")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(top["word"], top["tf_idf"] * 1000, color=color)
        ax.set_xlabel("Inverse Document Frequency (x1000)")
        ax.set_ylabel("Concept")
        ax.set_title(title)
        fig.tight_layout()
        _save(fig, f"fig6_{slug}.png")


# ---------------------------------------------------------------------------
# B1 - Sensitivity by duplication
# ---------------------------------------------------------------------------

def figure_b1_sensitivity(procdata: pd.DataFrame, procdata_long: pd.DataFrame):
    counts_unique = procdata.groupby("Year").size().reset_index(name="UniqueCount")
    counts_long = procdata_long.groupby("Year").size().reset_index(name="MultiCount")
    merged = counts_unique.merge(counts_long, on="Year", how="outer").fillna(0).sort_values("Year")
    
    with plt.style.context("ggplot"):
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
        
        # Area chart style
        ax.fill_between(merged["Year"], merged["UniqueCount"], merged["MultiCount"], color="#e74c3c", alpha=0.2, label="Multi-indexing Inflation")
        ax.fill_between(merged["Year"], 0, merged["UniqueCount"], color="#3498db", alpha=0.2)
        
        ax.plot(merged["Year"], merged["UniqueCount"], label="Unique Articles (Base)", color="#2980b9", marker="o", linewidth=2.5)
        ax.plot(merged["Year"], merged["MultiCount"], label="Multi-indexed (Expanded)", color="#c0392b", marker="s", linestyle="--", linewidth=2.5)
        
        ax.set_title("Methodological Sensitivity: Article Duplication via Multi-indexing", fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel("Publication Year", fontsize=12)
        ax.set_ylabel("Volume of Articles", fontsize=12)
        
        breaks = _five_year_breaks(merged["Year"])
        ax.set_xticks(breaks)
        
        # Value annotations for the first and last point
        for col, c in [("UniqueCount", "#2980b9"), ("MultiCount", "#c0392b")]:
            ax.text(merged["Year"].iloc[0], merged[col].iloc[0] + max(merged["MultiCount"])*0.02, str(int(merged[col].iloc[0])), color=c, fontweight='bold', ha='center')
            ax.text(merged["Year"].iloc[-1], merged[col].iloc[-1] + max(merged["MultiCount"])*0.02, str(int(merged[col].iloc[-1])), color=c, fontweight='bold', ha='center')

        ax.legend(loc="upper left", frameon=True, shadow=True, facecolor="white")
        fig.tight_layout()
        _save(fig, "fig_b1_sensitivity.png")

# ---------------------------------------------------------------------------
# B2 - Temporal and Economy Sensitivity
# ---------------------------------------------------------------------------
def figure_b2_temporal_economy(procdata_long: pd.DataFrame):
    df = _explode_countries(procdata_long.dropna(subset=["Macro_areas", "Country_affiliation"])).copy()
    k = df.groupby("_row_id")["Country_affiliation"].transform("count")
    df["w"] = 1.0 / k
    df["iso3"] = df["Country_affiliation"].map(GEOUNIT_TO_ISO3)
    df["economy"] = df["iso3"].map(ECONOMY_CLASS)
    
    df_early = df[df["Year"] <= 2017].copy()
    df_late = df[df["Year"] >= 2018].copy()
    
    # Custom colors for Macro Areas using config palette
    colors = {ma: MACRO_PALETTE.get(ma, "#333333") for ma in df["Macro_areas"].unique()}
    
    with plt.style.context("ggplot"):
        for name, sub_df in [("2010_2017", df_early), ("2018_2025", df_late)]:
            if sub_df.empty:
                continue
            weighted = sub_df.dropna(subset=["economy"]).groupby(["economy", "Macro_areas"])["w"].sum().reset_index(name="weight")
            weighted["economy"] = pd.Categorical(weighted["economy"], categories=ECON_LEVELS, ordered=True)
            pivot_df = weighted.pivot(index="economy", columns="Macro_areas", values="weight").fillna(0)
            
            fig, ax = plt.subplots(figsize=(12, 7), dpi=150)
            pivot_df.plot(kind="barh", stacked=True, ax=ax, color=[colors.get(c, "#333") for c in pivot_df.columns])
            
            ax.set_title(f"Knowledge Production by Economy ({name.replace('_', '-')})", fontsize=15, fontweight='bold', pad=15)
            ax.set_xlabel("Weighted Contribution (Fractional Counting)", fontsize=12)
            ax.set_ylabel("")
            
            # Clean up axes and spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(axis='y', labelsize=11)
            
            # Move legend outside
            ax.legend(title="Macro Area", bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
            
            fig.tight_layout()
            _save(fig, f"fig_b2_economy_{name}.png")

def table_b2_temporal_economy_proportions(procdata: pd.DataFrame):
    df_unique = _explode_countries(procdata.dropna(subset=['Country_affiliation'])).copy()
    k_unique = df_unique.groupby('_row_id')['Country_affiliation'].transform('count')
    df_unique['w'] = 1.0 / k_unique
    df_unique['iso3'] = df_unique['Country_affiliation'].map(GEOUNIT_TO_ISO3)
    df_unique['economy'] = df_unique['iso3'].map(ECONOMY_CLASS)

    df_early = df_unique[df_unique['Year'] <= 2017].copy()
    df_late = df_unique[df_unique['Year'] >= 2018].copy()

    def get_proportions(sub_df):
        if sub_df.empty: return pd.DataFrame()
        agg = sub_df.groupby(['economy', 'Country_affiliation'])['w'].sum().reset_index()
        agg = agg[agg['w'] > 0]
        total_w = agg['w'].sum()
        agg['Proportion (%)'] = (agg['w'] / total_w) * 100
        agg = agg.rename(columns={'economy': 'Economic Group', 'Country_affiliation': 'Country', 'w': 'Fractional Articles'})
        agg['Economic Group'] = pd.Categorical(agg['Economic Group'], categories=ECON_LEVELS, ordered=True)
        agg = agg.sort_values(['Economic Group', 'Fractional Articles'], ascending=[True, False]).reset_index(drop=True)
        return agg

    prop_early = get_proportions(df_early)
    prop_late = get_proportions(df_late)

    out_path = FIG_DIR.parent / 'Supplementary_Material_Country_Proportions.xlsx'
    with pd.ExcelWriter(out_path) as writer:
        prop_early.to_excel(writer, sheet_name='2010-2017', index=False)
        prop_late.to_excel(writer, sheet_name='2018-2025', index=False)
    print(f"  -> Saved {out_path.name}")

# ---------------------------------------------------------------------------
# B5 - Semantic Deficit (Figure 7)
# ---------------------------------------------------------------------------

def figure_b5_semantic_deficit(procdata: pd.DataFrame):
    lines = list(AGENDA_KEYWORDS.keys())
    counts = {line: 0 for line in lines}
    
    for txt in procdata["clean_abs"].dropna():
        for line, keywords in AGENDA_KEYWORDS.items():
            if any(kw in txt for kw in keywords):
                counts[line] += 1
                
    df = pd.DataFrame(list(counts.items()), columns=["AgendaLine", "Count"])
    df = df.sort_values("Count", ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(df["AgendaLine"], df["Count"], color="orange")
    ax.set_xlabel("Number of Articles containing at least one keyword")
    ax.set_title("Semantic Deficit by Agenda Line (Table 4)")
    
    for i, v in enumerate(df["Count"]):
        ax.text(v + max(df["Count"])*0.01, i, str(v), va='center')
        
    fig.tight_layout()
    _save(fig, "fig_b5_semantic_deficit.png")

# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_all(procdata: pd.DataFrame, procdata_long: pd.DataFrame):
    print("  -> figure_1")
    figure_1(procdata_long)
    print("  -> figure_2")
    figure_2(procdata_long)
    print("  -> figure_3")
    figure_3(procdata, procdata_long)
    print("  -> figure_4")
    figure_4(procdata_long)
    print("  -> figure_5")
    figure_5(procdata)
    print("  -> figure_6")
    figure_6(procdata_long)

    print("  -> figure_b1")
    figure_b1_sensitivity(procdata, procdata_long)
    print("  -> figure_b2")
    figure_b2_temporal_economy(procdata_long)
    print("  -> table_b2_proportions")
    table_b2_temporal_economy_proportions(procdata)
    print("  -> figure_b5")
    figure_b5_semantic_deficit(procdata)
    print("  -> All done in run_all")

