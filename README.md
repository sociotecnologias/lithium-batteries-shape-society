# Bibliometric Analysis

Python pipeline for generating the bibliometric and semantic analysis for the paper "How Lithium Batteries Shape Society? From Tech Tools to Socio-Technical Devices", published in *International Journal of Energy Research* ([doi:10.1155/er/5162683](https://doi.org/10.1155/er/5162683)).

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

## Citation

If you use this code or its outputs, please cite the paper:

> Ojeda-Pereira, Iván, Herrera-León, Sebastián, Santibáñez Ferreira, Javier, Kraslawski, Andrzej, Campos-Medina, Fernando, Cassola, José, How Lithium Batteries Shape Society? From Tech Tools to Socio-Technical Devices, *International Journal of Energy Research*, 2026, 5162683, 19 pages, 2026. https://doi.org/10.1155/er/5162683

BibTeX:

```bibtex
@article{ojedapereira2026lithium,
  author   = {Ojeda-Pereira, Iván and Herrera-León, Sebastián and Santibáñez Ferreira, Javier and Kraslawski, Andrzej and Campos-Medina, Fernando and Cassola, José},
  title    = {How Lithium Batteries Shape Society? From Tech Tools to Socio-Technical Devices},
  journal  = {International Journal of Energy Research},
  volume   = {2026},
  number   = {1},
  pages    = {5162683},
  year     = {2026},
  doi      = {10.1155/er/5162683},
  url      = {https://onlinelibrary.wiley.com/doi/abs/10.1155/er/5162683},
  keywords = {bibliometric, lithium batteries, science and technology studies, social sciences, transdisciplinary engineering}
}
```
