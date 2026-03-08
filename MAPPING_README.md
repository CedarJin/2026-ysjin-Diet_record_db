# FNDDS Record Mapping Pipeline

This pipeline maps dietary records to the FNDDS (Food and Nutrient Database for Dietary Studies) database using FTS5 (Full-Text Search 5) for recall and sentence-transformers embeddings for reranking.

## Overview

The pipeline consists of 6 steps:

1. **Create FNDDS Food Index**: Normalizes food descriptions from FNDDS database and creates a searchable index table with FTS5 virtual table
2. **Precompute FNDDS Embeddings**: Uses `sentence-transformers/all-MiniLM-L6-v2` to encode normalized FNDDS descriptions and saves food IDs + embeddings to a numpy file (`db/fndds/fndds_embeddings.npz`) for fast lookup during matching
3. **Parse Records**: Extracts food items from reviewed HTML records and generates JSON files
4. **GPT-5.2 Normalization**: Uses GPT-5.2 to normalize food descriptions and extract structured information (core_food, modifiers, process_method)
5. **Embedding Similarity Matching**: Queries using **normalized description**, ranks all FNDDS candidates by embedding similarity, and returns top 10 per food
6. **Visualization**: Generates HTML files showing **top 10** candidates per food with similarity score

## Installation

Install required dependencies (from project root):

```bash
uv sync
uv add sentence-transformers numpy
```

All dependencies are defined in `pyproject.toml`. For embedding rerank you need `sentence-transformers` and `numpy`.

### OpenAI API Setup

The normalization step requires an OpenAI API key. Set it up as follows:

1. Create a `.env` file in the project root (if it doesn't exist)
2. Add your OpenAI API key:
   ```
   OPENAI_API_KEY=sk-...your-actual-key-here...
   ```

The pipeline will skip normalization if the API key is not found, but matching will still work using basic text normalization.

## Usage

Run the complete pipeline:

```bash
uv run python fndds_record_mapping.py
```

## Output Structure

The pipeline generates files in `record/Mapping/`:

- `{record_id}_record.json`: Normalized record data with metadata and food items (includes core_food, modifiers, process_method, normalized_description)
- `{record_id}_matches.json`: Match results with **top 10** per food (each item has `food_code`, `main_food_description`, `similarity_score` [embedding similarity])
- `{record_id}_matches.html`: Visualization HTML showing **top 10** candidates per food with similarity score
- `db/fndds/fndds_embeddings.npz`: Precomputed FNDDS embeddings (food_codes + embeddings) from Step 2

### Normalized JSON Structure

Each food item in `{record_id}_record.json` includes:
- `food_description`: Original food description
- `normalized_description`: Description with quantity words removed
- `core_food`: Main food item (e.g., "coffee", "turkey", "potato")
- `modifiers`: List of descriptive modifiers (e.g., ["espresso machine", "deli", "thin slices"])
- `process_method`: Cooking/preparation method if mentioned (e.g., "baked", "grilled", null)
- `amount`: Amount value
- `unit`: Unit of measurement

## Normalization

### GPT-5.2 Normalization

Food descriptions are normalized using GPT-5.2 to extract structured information:

- **core_food**: The main food item without modifiers (always in singular form)
- **modifiers**: Descriptive words or phrases that modify the core food (in singular form where applicable)
- **process_method**: Cooking or preparation method (if mentioned)
- **normalized_description**: Description with quantity words removed and converted to singular form (since amount and unit are stored separately)

**Important**: All output fields use singular forms. Plurals are converted to singular (e.g., "dates" → "date", "slices" → "slice", "crackers" → "cracker").

Example:
- Input: "4 thin slices deli turkey"
- Output:
  - `core_food`: "turkey"
  - `modifiers`: ["thin slice", "deli"]
  - `process_method`: null
  - `normalized_description`: "thin slice deli turkey"

Example with plural:
- Input: "5 dates"
- Output:
  - `core_food`: "date"
  - `modifiers`: []
  - `process_method`: null
  - `normalized_description`: "date"

### Basic Text Normalization (v1.0)

For FNDDS database matching, food descriptions are further normalized:
- **NFS / NS:** Capitalized whole-word "NFS" → "not further specified"; "NS" → "not specified" (before lowercasing)
- **Percentages:** Punctuation is removed *except* "." and "%", so values like "3.25%" are kept
- Convert to lowercase
- Single space between words

Examples:
- Input: "Milk, whole, 3.25% milkfat"  
  Normalized: "milk whole 3.25% milkfat"
- Input: "Coffee, NFS"  
  Normalized: "coffee not further specified"
- Input: "Tea, NS"  
  Normalized: "tea not specified"

## Database Schema

The pipeline creates a `fndds_food_index` table in the FNDDS database with:
- `food_code`: TEXT PRIMARY KEY
- `main_food_description`: Original description
- `normalized_food_description`: Normalized version
- `source_version`: "FNDDS 2021-2023"
- `normalize_version`: "v1.0"

## Embedding Similarity Ranking

**Ranking Strategy:** The pipeline uses **embedding similarity only** (no FTS5 pre-filtering). The query string is **normalized_description**; it is encoded with `sentence-transformers/all-MiniLM-L6-v2`. Cosine similarity is computed between the query embedding and **ALL** precomputed FNDDS candidate embeddings. Candidates are ranked by similarity and **top 10** are returned per food. Each result includes:
- `similarity_score`: embedding cosine similarity (0.0 to 1.0, higher is better)

**Precomputed embeddings:** Step 2 writes `db/fndds/fndds_embeddings.npz` (food_codes + embeddings) so similarity can be computed efficiently for all candidates without recomputing embeddings.

## Code Standards

All code follows Python best practices:
- Type hints for all functions
- Comprehensive docstrings
- Dataclasses for data structures
- Error handling
- English comments throughout
