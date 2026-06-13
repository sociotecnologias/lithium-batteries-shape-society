# Bibliometric Analysis

Python pipeline for generating the bibliometric and semantic analysis for the manuscript "Main_International Journal of Energy_V4".

## Layout

```
.
├── databases/                   # reference Excel files (Macroareas, stopwords, etc.)
├── input/                       # raw data (e.g. scopus_20_8_25.csv)
├── outputs/
│   ├── figures/                 # generated PNGs for the paper
│   ├── tidy/                    # processed datasets (procdata_final, procdata_long_final)
│   └── Supplementary_Material_Country_Proportions.xlsx
├── config.py                    # dictionaries and parameters
├── preparation.py               # data cleaning and normalization
├── analysis.py                  # statistical analysis and figure generation
├── main.py                      # entry point
└── requirements.txt
```

## Setup

It is highly recommended to use a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run the Pipeline

1. Ensure your raw scopus export (`scopus_20_8_25.csv`) is inside `input/`.
2. Execute the main script:

```powershell
python main.py
```

## Output

The script automatically generates all figures and tables used in the revised manuscript:
* **`outputs/tidy/procdata_final.xlsx`**: Wide table, one row per article.
* **`outputs/tidy/procdata_long_final.xlsx`**: Long table, exploded by Macro Area.
* **`outputs/figures/*.png`**: All visual assets including:
  * Figure 1 & 2 (Growth and Citations)
  * Figure 3 (Maps and Bar charts by economy)
  * Figure 4, 5 & 6 (Tokens, Bigram Networks, TF-IDF)
  * Figure B1 (Sensitivity to Multi-indexing duplication)
  * Figure B2 (Economy Subgroup Sensitivity Sunburst/Bars)
  * Figure B5 (Semantic Deficit Radar/Bars)
  * Figure B6 (Interdisciplinarity Networks)
* **`outputs/Supplementary_Material_Country_Proportions.xlsx`**: The exact fractional counting production table explicitly mapping every country to its economic group (2010-2017 vs 2018-2025).
