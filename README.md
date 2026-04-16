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

If your OpenAI project is region-scoped and you see an `incorrect regional hostname` or `incorrect_hostname` error, also add:

```bash
OPENAI_BASE_URL=https://us.api.openai.com/v1
```

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
http://127.0.0.1:3004/record/OCR/R0001_OCR.html?serverWindowId=de70c997-75e6-4ee5-b9c9-a0560fa9df04

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
6. Maps the original amount/unit to one portion code for GPT-selected FNDDS food, then converts amount to grams
7. Generates visualization HTML: GPT selection shown first, then top 10 with similarity scores and portion/gram conversion
8. Calculates FNDDS nutrient amounts per food and daily nutrient totals into `results/fndds`
9. Generates readable FNDDS nutrient HTML reports in `results/fndds`

```bash
uv run script/fndds_record_mapping.py
```
Mapping results example:
http://127.0.0.1:3004/record/Mapping/R0001_matches.html?serverWindowId=7208c31c-55d8-40c5-bb15-34c590c949a0

## Glycopedia monosaccharide mapping pipeline

This step keeps Glycopedia separate from FNDDS. It builds `db/glycopedia/glycopedia.db`, creates Glycopedia embeddings, creates FNDDS ingredient-to-Glycopedia top candidates, then uses existing FNDDS record matches to estimate ingredient-level monosaccharide intake.

By default, GPT is not used. Embedding candidates are shown as suggestions only and are marked `needs_review_embedding_only`; they are not used for final monosaccharide totals until selected by GPT or review.

```bash
uv run script/glycopedia_mapping.py
```

To use GPT for final ingredient-to-Glycopedia selection/rejection:

```bash
uv run script/glycopedia_mapping.py --skip-db --skip-embeddings --skip-candidates --use-gpt
```

Outputs:

- `db/glycopedia/glycopedia.db`
- `db/glycopedia/glycopedia_embeddings.npz`
- `record/GlycopediaMapping/fndds_ingredient_glycopedia_candidates.json`
- `record/GlycopediaMapping/{record_id}_glycopedia_matches.json/html`
- `results/glycopedia/{record_id}_glycopedia_monosaccharides.json/html`

## Polyphenol mapping pipeline

This step keeps Phenol-Explorer/polyphenol data separate from FNDDS and Glycopedia. It builds `db/polyphenol/polyphenol.db`, creates stable local food IDs for Phenol-Explorer foods, creates food embeddings, creates used FNDDS ingredient-to-polyphenol food top candidates, then uses existing FNDDS record matches to estimate ingredient-level individual polyphenol intake.

Version 1 is conservative: embedding candidates are suggestions only by default. They are not used for final polyphenol totals until selected by GPT or later review. It does not do missing compound imputation, component expansion, weighted composites, or cooked/raw yield adjustment.

```bash
uv run script/polyphenol_mapping.py
```

To use GPT for final ingredient-to-polyphenol food selection/rejection:

```bash
uv run script/polyphenol_mapping.py --skip-db --skip-embeddings --skip-candidates --use-gpt
```

Outputs:

- `db/polyphenol/polyphenol.db`
- `db/polyphenol/polyphenol_embeddings.npz`
- `record/PolyphenolMapping/fndds_ingredient_polyphenol_candidates.json`
- `record/PolyphenolMapping/{record_id}_polyphenol_matches.json/html`
- `results/polyphenol/{record_id}_polyphenol_intakes.json/html`

## Next steps/goals
1. Convert the intermediate results to Excel sheets for human verification
2. Use GPT selection for ingredientization and following mapping to glycan/polyphenol databases
3. Use NUTRIBENCH dataset + USDA dataset to fine tune semantic search model all-miniLM-L6-v2 for better semantic similarity calculation
4. Try open-sourced models for OCR and normalization
5. Localized the LLM for reranking
6. Test system performance with more data - performance metrics?
7. Release as an open-sourced tool for research
