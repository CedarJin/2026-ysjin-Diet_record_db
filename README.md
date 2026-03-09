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

## Records Optical Character Recognition (OCR)
This step uses GPT-4o to parse dietary records page by page.
```bash
uv run script/parse_dietary_record.py
```
Example output:
http://127.0.[0].1:3004/record/OCR/R0001_OCR.html?serverWindowId=de70c997-75e6-4ee5-b9c9-a0560fa9df04

## FNDDS database processing
This step reads database files to CSV files and constructs SQLite database from FNDDS CSV files.
```bash
uv run script/read_sas_file.py
uv run script/build_fndds_db.py
```

## FNDDS database mapping pipeline

1. Normalizes FNDDS food descriptions and creates fndds_food_index table
2. Precomputes FNDDS candidate embeddings (sentence-transformers/all-MiniLM-L6-v2) and saves as numpy file
3. Parses reviewed HTML records and generates JSON objects
4. Normalizes food descriptions using GPT-5.2 to extract structured information
5. Matches records to FNDDS by embedding similarity, returns top 10; GPT selects one final candidate per food by meaning
6. Generates visualization HTML: GPT selection shown first, then top 10 with similarity scores

```bash
uv run script/fndds_record_mapping.py
```
Mapping results example:
http://127.0.[0].1:3004/record/Mapping/R0001_matches.html?serverWindowId=7208c31c-55d8-40c5-bb15-34c590c949a0

## Next steps
1. Incorporate the portion size in the results
2. Convert the intermediate results to Excel sheets for human verification
3. Use GPT selection for ingredientization and following mapping to glycan/polyphenol databases
4. Calculate the nutrient contents
5. Use NUTRIBENCH dataset + USDA dataset to fine tune semantic search model all-miniLM-L6-v2 for better semantic calculation
6. Try open-sourced models for OCR and normalization
7. Localized the LLM for reranking
8. Test system performance with more data
9. Release as a open-sourced tool for research
