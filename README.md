# 2026-ysjin-Diet_record_db
The project aims to create a LLM-assisted workflow for dietary records parsing and mapping to databases that necessary for precise functional microbiome analysis.

## Prepare databases
FNNDS database was downloaded from https://www.ars.usda.gov/northeast-area/beltsville-md-bhnrc/beltsville-human-nutrition-research-center/food-surveys-research-group/docs/fndds-download-databases/

Glycopedia database was downloaded from https://github.com/quarksome/Food-Glycopedia

Phenol-Explorer database was downloaed from https://security.ucop.edu/policies/institutional-information-and-it-resource-classification.html

## Environment deployment
```bash
uv sync
source .venv/bin/activate
```
Add your openAI api key to .env.

## Records Overview

Based on `record/Raw/record_meta.xlsx` (6 records):

| Metric | Count |
| --- | ---: |
| Total records | 6 |
| PAT diet records | 3 |
| HAB diet records | 3 |
| Pre records | 6 |
| Post records | 0 |
| Records with flag | 0 |
