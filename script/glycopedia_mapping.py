#!/usr/bin/env python3
"""
Glycopedia monosaccharide mapping pipeline.

This script is intentionally separate from script/fndds_record_mapping.py.

Workflow:
1. Build a Glycopedia SQLite database from the Excel files.
2. Build a Glycopedia food index and embeddings.
3. Build FNDDS ingredient -> Glycopedia top-k embedding candidates.
4. Use existing FNDDS record matches to ingredientize foods.
5. Map used FNDDS ingredients to Glycopedia entries, with optional GPT final selection.
6. Calculate monosaccharide intake and write JSON/HTML reports.
"""

import argparse
import html
import json
import os
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from dotenv import load_dotenv
    from openai import OpenAI
    load_dotenv()
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


FNDDS_DB_PATH = Path("db/fndds/fndds_2021_2023.db")
FNDDS_MATCH_DIR = Path("record/FNDDSMapping")

GLYCOPEDIA_DIR = Path("db/glycopedia")
GLYCOPEDIA_METADATA_XLSX = GLYCOPEDIA_DIR / "10_glycopedia_metadata_021022.xlsx"
GLYCOPEDIA_MONO_XLSX = GLYCOPEDIA_DIR / "NIH_all_mono_newFGv3.xlsx"
GLYCOPEDIA_DB_PATH = GLYCOPEDIA_DIR / "glycopedia.db"
GLYCOPEDIA_EMBEDDINGS_PATH = GLYCOPEDIA_DIR / "glycopedia_embeddings.npz"

GLYCOPEDIA_MAPPING_DIR = Path("record/GlycopediaMapping")
GLYCOPEDIA_RESULTS_DIR = Path("results/glycopedia")

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K_CANDIDATES = 10
SOURCE_VERSION = "Food-Glycopedia 2022-02-10"
NORMALIZE_VERSION = "v1.0"

MONOSACCHARIDE_COLUMNS = [
    "Glucose",
    "Galactose",
    "Fructose",
    "Xylose",
    "Arabinose",
    "Fucose",
    "Rhamnose",
    "GlcA",
    "GalA",
    "GlcNAc",
    "GalNAc",
    "Mannose",
    "Allose",
    "Ribose",
]

GLYCOPEDIA_FOOD_GROUPS = {
    1: "milk and milk products",
    2: "meat, poultry, fish, and mixtures",
    3: "eggs",
    4: "beans, peas, legumes, nuts, and seeds",
    5: "grain products",
    6: "fruits",
    7: "vegetables",
    8: "fats, oils, and salad dressings",
    9: "sugars, sweets, and beverages",
}


def create_openai_client() -> "OpenAI":
    """Create an OpenAI client, honoring an optional regional base URL."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key)


def normalize_text(text: Any) -> str:
    """Normalize food text for search/embedding."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    text = str(text)
    text = re.sub(r"\bNFS\b", "not further specified", text)
    text = re.sub(r"\bNS\b", "not specified", text)
    text = text.replace("+", " and ")
    text = text.lower()
    text = re.sub(r"[^\w\s.%/]", " ", text)
    text = re.sub(r"(?<!\d)\.(?!\d)", " ", text)
    text = re.sub(r"(?<!\d)/(?!\d)", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def html_escape(value: Any) -> str:
    """Escape a value for HTML."""
    if value is None:
        return ""
    return html.escape(str(value))


def format_number(value: Any, digits: int = 4) -> str:
    """Format numeric values compactly."""
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return ""


def safe_float(value: Any) -> Optional[float]:
    """Convert a value to float, returning None on failure."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def create_index_text(row: pd.Series) -> str:
    """Create a concise but informative Glycopedia embedding text."""
    parts = [
        row.get("food_description"),
        row.get("product_name"),
        row.get("processing"),
        row.get("food_group_description"),
        row.get("single_multi_ingredient"),
        row.get("ingredient_list"),
        row.get("additional_descriptions"),
        row.get("preparation_method"),
    ]
    return "; ".join(str(p) for p in parts if p is not None and not pd.isna(p) and str(p).strip())


def build_glycopedia_db(db_path: Path = GLYCOPEDIA_DB_PATH) -> Path:
    """Build Glycopedia SQLite database from metadata and mono Excel files."""
    print("=" * 80)
    print("Building Glycopedia SQLite database")
    print("=" * 80)

    metadata = pd.read_excel(
        GLYCOPEDIA_METADATA_XLSX,
        sheet_name="10_glycopedia_metadata_020222",
    )
    mono = pd.read_excel(GLYCOPEDIA_MONO_XLSX, sheet_name="g per 100g FW")

    metadata["food_group_description"] = metadata["food_group"].map(GLYCOPEDIA_FOOD_GROUPS)
    mono = mono.rename(columns={"sample_ID": "glycan_id", "food_group": "mono_food_group"})

    index_df = metadata.merge(mono[["glycan_id"]], on="glycan_id", how="inner")
    index_df["index_text"] = index_df.apply(create_index_text, axis=1)
    index_df["normalized_index_text"] = index_df["index_text"].map(normalize_text)
    index_df["source_version"] = SOURCE_VERSION
    index_df["normalize_version"] = NORMALIZE_VERSION

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        metadata.to_sql("glycopedia_metadata", conn, if_exists="replace", index=False)
        mono.to_sql("glycopedia_mono", conn, if_exists="replace", index=False)
        index_cols = [
            "glycan_id",
            "food_description",
            "product_name",
            "processing",
            "food_group",
            "food_group_description",
            "single_multi_ingredient",
            "ingredient_list",
            "additional_descriptions",
            "preparation_method",
            "index_text",
            "normalized_index_text",
            "source_version",
            "normalize_version",
        ]
        index_df[index_cols].to_sql("glycopedia_food_index", conn, if_exists="replace", index=False)
        cur = conn.cursor()
        cur.execute("CREATE INDEX IF NOT EXISTS idx_glycopedia_index_glycan_id ON glycopedia_food_index(glycan_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_glycopedia_mono_glycan_id ON glycopedia_mono(glycan_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_glycopedia_metadata_glycan_id ON glycopedia_metadata(glycan_id)")
        conn.commit()
    finally:
        conn.close()

    print(f"metadata rows: {len(metadata)}")
    print(f"mono rows: {len(mono)}")
    print(f"indexed rows with mono data: {len(index_df)}")
    print(f"wrote {db_path}")
    return db_path


def build_glycopedia_embeddings(
    db_path: Path = GLYCOPEDIA_DB_PATH,
    embeddings_path: Path = GLYCOPEDIA_EMBEDDINGS_PATH,
    model_name: str = EMBEDDING_MODEL_NAME,
) -> Path:
    """Build and save Glycopedia food index embeddings."""
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        raise RuntimeError("sentence-transformers is required for Glycopedia embeddings")

    print("=" * 80)
    print("Building Glycopedia embeddings")
    print("=" * 80)

    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            "SELECT glycan_id, normalized_index_text FROM glycopedia_food_index ORDER BY glycan_id"
        ).fetchall()
    finally:
        conn.close()

    glycan_ids = np.array([int(row[0]) for row in rows], dtype=np.int64)
    texts = [row[1] or " " for row in rows]
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(embeddings_path, glycan_ids=glycan_ids, embeddings=embeddings)
    print(f"wrote {embeddings_path}")
    return embeddings_path


def load_glycopedia_embeddings(embeddings_path: Path = GLYCOPEDIA_EMBEDDINGS_PATH) -> Tuple[np.ndarray, np.ndarray]:
    """Load Glycopedia embeddings as glycan_ids and matrix."""
    data = np.load(embeddings_path, allow_pickle=True)
    return data["glycan_ids"], np.asarray(data["embeddings"], dtype=np.float32)


def get_glycopedia_metadata_by_id(db_path: Path = GLYCOPEDIA_DB_PATH) -> Dict[int, Dict[str, Any]]:
    """Load Glycopedia index metadata keyed by glycan_id."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("SELECT * FROM glycopedia_food_index").fetchall()
        return {int(row["glycan_id"]): dict(row) for row in rows}
    finally:
        conn.close()


def get_unique_fndds_ingredients(fndds_db_path: Path = FNDDS_DB_PATH) -> List[Dict[str, str]]:
    """Return unique FNDDS ingredient code/description pairs."""
    conn = sqlite3.connect(str(fndds_db_path))
    try:
        rows = conn.execute(
            """
            SELECT Ingredient_code, Ingredient_description
            FROM fnddsingred
            WHERE Ingredient_code IS NOT NULL
            GROUP BY Ingredient_code, Ingredient_description
            ORDER BY CAST(Ingredient_code AS INTEGER)
            """
        ).fetchall()
    finally:
        conn.close()
    return [
        {
            "ingredient_code": str(code),
            "ingredient_description": description or "",
            "normalized_ingredient_description": normalize_text(description),
        }
        for code, description in rows
    ]


def cosine_top_k(
    query_embeddings: np.ndarray,
    candidate_embeddings: np.ndarray,
    top_k: int,
) -> List[List[Tuple[int, float]]]:
    """Return candidate matrix indices and cosine scores for each query embedding."""
    q_norm = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
    c_norm = np.linalg.norm(candidate_embeddings, axis=1, keepdims=True)
    sims = (query_embeddings @ candidate_embeddings.T) / (q_norm * c_norm.T + 1e-9)
    out: List[List[Tuple[int, float]]] = []
    for row in sims:
        order = np.argsort(-row)[:top_k]
        out.append([(int(idx), float(row[idx])) for idx in order])
    return out


def build_fndds_ingredient_glycopedia_candidates(
    db_path: Path = GLYCOPEDIA_DB_PATH,
    embeddings_path: Path = GLYCOPEDIA_EMBEDDINGS_PATH,
    top_k: int = TOP_K_CANDIDATES,
    model_name: str = EMBEDDING_MODEL_NAME,
) -> Path:
    """Build top-k Glycopedia candidates for every unique FNDDS ingredient."""
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        raise RuntimeError("sentence-transformers is required for candidate search")

    print("=" * 80)
    print("Building FNDDS ingredient -> Glycopedia candidates")
    print("=" * 80)

    ingredients = get_unique_fndds_ingredients()
    glycan_ids, glyco_embeddings = load_glycopedia_embeddings(embeddings_path)
    glyco_meta = get_glycopedia_metadata_by_id(db_path)
    model = SentenceTransformer(model_name)
    query_texts = [item["normalized_ingredient_description"] or " " for item in ingredients]
    query_embeddings = model.encode(query_texts, show_progress_bar=True, convert_to_numpy=True)
    query_embeddings = np.asarray(query_embeddings, dtype=np.float32)
    ranked = cosine_top_k(query_embeddings, glyco_embeddings, top_k)

    candidate_rows = []
    candidate_json: Dict[str, Any] = {
        "source": "FNDDS ingredient descriptions vs Glycopedia food index",
        "embedding_model": model_name,
        "top_k": top_k,
        "ingredients": [],
    }
    for ingredient, top_matches in zip(ingredients, ranked):
        candidates = []
        for rank, (idx, score) in enumerate(top_matches, start=1):
            glycan_id = int(glycan_ids[idx])
            meta = glyco_meta.get(glycan_id, {})
            candidate = {
                "rank": rank,
                "glycan_id": glycan_id,
                "food_description": meta.get("food_description"),
                "product_name": meta.get("product_name"),
                "processing": meta.get("processing"),
                "food_group": meta.get("food_group"),
                "food_group_description": meta.get("food_group_description"),
                "single_multi_ingredient": meta.get("single_multi_ingredient"),
                "ingredient_list": meta.get("ingredient_list"),
                "similarity_score": score,
            }
            candidates.append(candidate)
            candidate_rows.append({
                "ingredient_code": ingredient["ingredient_code"],
                "ingredient_description": ingredient["ingredient_description"],
                "rank": rank,
                "glycan_id": glycan_id,
                "glycopedia_food_description": meta.get("food_description"),
                "similarity_score": score,
            })
        candidate_json["ingredients"].append({
            **ingredient,
            "candidates": candidates,
        })

    conn = sqlite3.connect(str(db_path))
    try:
        pd.DataFrame(candidate_rows).to_sql(
            "fndds_ingredient_glycopedia_candidates",
            conn,
            if_exists="replace",
            index=False,
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_fndds_glyco_candidates_ingredient ON fndds_ingredient_glycopedia_candidates(ingredient_code)"
        )
        conn.commit()
    finally:
        conn.close()

    GLYCOPEDIA_MAPPING_DIR.mkdir(parents=True, exist_ok=True)
    out_path = GLYCOPEDIA_MAPPING_DIR / "fndds_ingredient_glycopedia_candidates.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(candidate_json, f, indent=2, ensure_ascii=False)

    print(f"ingredients: {len(ingredients)}")
    print(f"wrote {out_path}")
    return out_path


def load_candidate_cache(candidate_path: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Load candidate JSON as ingredient_code -> candidate list."""
    with open(candidate_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        item["ingredient_code"]: item.get("candidates", [])
        for item in data.get("ingredients", [])
    }


def get_fndds_food_ingredients(conn: sqlite3.Connection, food_code: str) -> List[Dict[str, Any]]:
    """Return FNDDS ingredient rows for one FNDDS food code."""
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT Ingredient_code, Ingredient_description, Ingredient_weight
        FROM fnddsingred
        WHERE CAST(Food_code AS TEXT) = ?
        ORDER BY Seq_num
        """,
        (str(food_code),),
    ).fetchall()
    return [dict(row) for row in rows]


def calculate_ingredient_gram_allocations(
    food_grams: float,
    ingredients: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Allocate consumed FNDDS food grams across ingredient rows.

    FNDDS Ingredient_weight rows can sum to values other than 100 for recipe foods,
    so Glycopedia ingredient-level intake uses normalized fractions within each
    FNDDS food instead of Ingredient_weight / 100.
    """
    weights = [
        safe_float(ingredient.get("Ingredient_weight"))
        for ingredient in ingredients
    ]
    total_weight = sum(weight for weight in weights if weight is not None and weight > 0)

    allocated = []
    for ingredient, weight in zip(ingredients, weights):
        if weight is None or weight <= 0 or total_weight <= 0:
            fraction = None
            ingredient_grams = None
        else:
            fraction = weight / total_weight
            ingredient_grams = round(food_grams * fraction, 6)
        allocated.append({
            **ingredient,
            "ingredient_weight_raw": weight,
            "ingredient_weight_sum_for_food": total_weight if total_weight > 0 else None,
            "ingredient_weight_fraction": fraction,
            "ingredient_grams": ingredient_grams,
        })
    return allocated


def ingredientize_match_file(match_file: Path, fndds_db_path: Path = FNDDS_DB_PATH) -> List[Dict[str, Any]]:
    """Convert one FNDDS match file into ingredient-level gram contributions."""
    with open(match_file, "r", encoding="utf-8") as f:
        match_data = json.load(f)

    conn = sqlite3.connect(str(fndds_db_path))
    try:
        ingredient_items = []
        for food_idx, food_match in enumerate(match_data.get("food_matches", []), start=1):
            food_code = food_match.get("gpt_selected_food_code")
            food_grams = safe_float(food_match.get("grams"))
            if not food_code or food_grams is None:
                continue
            ingredients = get_fndds_food_ingredients(conn, food_code)
            for ingredient in calculate_ingredient_gram_allocations(food_grams, ingredients):
                ingredient_items.append({
                    "record_id": match_data.get("record_id"),
                    "food_index": food_idx,
                    "fndds_food_code": food_code,
                    "record_food_description": food_match.get("food_description"),
                    "fndds_food_grams": food_grams,
                    "ingredient_code": str(ingredient.get("Ingredient_code")),
                    "ingredient_description": ingredient.get("Ingredient_description"),
                    "ingredient_weight_raw": ingredient.get("ingredient_weight_raw"),
                    "ingredient_weight_sum_for_food": ingredient.get("ingredient_weight_sum_for_food"),
                    "ingredient_weight_fraction": ingredient.get("ingredient_weight_fraction"),
                    "ingredient_grams": ingredient.get("ingredient_grams"),
                })
    finally:
        conn.close()
    return ingredient_items


def load_glycopedia_mono_by_id(db_path: Path = GLYCOPEDIA_DB_PATH) -> Dict[int, Dict[str, float]]:
    """Load mono values keyed by glycan_id."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("SELECT * FROM glycopedia_mono").fetchall()
    finally:
        conn.close()

    out = {}
    for row in rows:
        glycan_id = int(row["glycan_id"])
        out[glycan_id] = {
            mono: float(row[mono]) if row[mono] is not None else 0.0
            for mono in MONOSACCHARIDE_COLUMNS
        }
    return out


def select_glycopedia_match_with_gpt(
    ingredient_description: str,
    candidates: List[Dict[str, Any]],
    client: Optional["OpenAI"],
) -> Dict[str, Any]:
    """Ask GPT to select one Glycopedia candidate or reject all candidates."""
    if not candidates or client is None:
        best = candidates[0] if candidates else {}
        return {
            "selected_glycan_id": None,
            "match_status": "needs_review_embedding_only" if best else "no_candidates",
            "reason": "No GPT selection was run; top embedding candidate is shown as a suggestion only." if best else "No Glycopedia candidates found.",
        }

    candidates_text = "\n".join(
        "- glycan_id: {glycan_id} | {food_description} | product: {product_name} | "
        "processing: {processing} | group: {food_group_description} | ingredients: {ingredient_list} "
        "| similarity: {similarity_score:.4f}".format(**candidate)
        for candidate in candidates
    )
    prompt = f"""You are mapping an FNDDS ingredient to a Glycopedia food sample for monosaccharide intake estimation.

FNDDS ingredient description:
{ingredient_description}

Glycopedia candidates:
{candidates_text}

Choose one candidate only if it is a plausible edible-form match. Reject candidates when the form is wrong, such as brewed coffee vs coffee grounds, or when no candidate is close enough.

Return ONLY valid JSON:
{{
  "selected_glycan_id": 123 or null,
  "match_status": "matched" or "no_confident_match",
  "reason": "short reason"
}}"""
    try:
        response = client.chat.completions.create(
            model="gpt-5.2",
            messages=[
                {"role": "system", "content": "You select or reject Glycopedia matches for food composition mapping. Return only JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_completion_tokens=250,
        )
        content = (response.choices[0].message.content or "").strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        result = json.loads(content.strip())
    except Exception as exc:
        best = candidates[0]
        return {
            "selected_glycan_id": best.get("glycan_id"),
            "match_status": "needs_review_embedding_only",
            "reason": f"GPT selection failed: {exc}",
        }

    selected = result.get("selected_glycan_id")
    valid_ids = {int(candidate["glycan_id"]) for candidate in candidates}
    if selected is not None:
        try:
            selected = int(selected)
        except (TypeError, ValueError):
            selected = None
    if selected not in valid_ids:
        selected = None
    return {
        "selected_glycan_id": selected,
        "match_status": result.get("match_status") or ("matched" if selected else "no_confident_match"),
        "reason": result.get("reason", ""),
    }


def build_used_ingredient_mapping(
    ingredient_items: List[Dict[str, Any]],
    candidate_cache: Dict[str, List[Dict[str, Any]]],
    use_gpt: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """Build selected Glycopedia mapping for ingredients used in current records."""
    client = None
    if use_gpt:
        if not OPENAI_AVAILABLE:
            raise RuntimeError("OpenAI package is not available")
        client = create_openai_client()

    unique_ingredients = {}
    for item in ingredient_items:
        unique_ingredients.setdefault(item["ingredient_code"], item["ingredient_description"])

    mapping = {}
    for ingredient_code, description in sorted(unique_ingredients.items(), key=lambda x: x[0]):
        candidates = candidate_cache.get(ingredient_code, [])
        selected = select_glycopedia_match_with_gpt(description, candidates, client)
        selected_candidate = next(
            (candidate for candidate in candidates if candidate.get("glycan_id") == selected.get("selected_glycan_id")),
            None,
        )
        if selected_candidate is None and selected.get("match_status") == "needs_review_embedding_only" and candidates:
            selected_candidate = candidates[0]
        mapping[ingredient_code] = {
            "ingredient_code": ingredient_code,
            "ingredient_description": description,
            "selected_glycan_id": selected.get("selected_glycan_id"),
            "match_status": selected.get("match_status"),
            "reason": selected.get("reason"),
            "selected_candidate": selected_candidate,
            "candidates": candidates,
        }
    return mapping


def calculate_record_monosaccharides(
    record_id: str,
    ingredient_items: List[Dict[str, Any]],
    ingredient_mapping: Dict[str, Dict[str, Any]],
    mono_by_id: Dict[int, Dict[str, float]],
) -> Dict[str, Any]:
    """Calculate Glycopedia monosaccharide intake for one record."""
    ingredient_results = []
    totals = {mono: 0.0 for mono in MONOSACCHARIDE_COLUMNS}

    for item in ingredient_items:
        mapping = ingredient_mapping.get(item["ingredient_code"], {})
        glycan_id = mapping.get("selected_glycan_id")
        mono_values = mono_by_id.get(int(glycan_id), {}) if glycan_id is not None else {}
        ingredient_grams = safe_float(item.get("ingredient_grams"))
        mono_intake = {}
        for mono in MONOSACCHARIDE_COLUMNS:
            per_100g = safe_float(mono_values.get(mono)) or 0.0
            amount = None if ingredient_grams is None or glycan_id is None else round(per_100g * ingredient_grams / 100.0, 6)
            mono_intake[mono] = {
                "glycopedia_g_per_100g_fw": per_100g if glycan_id is not None else None,
                "amount_g": amount,
            }
            if amount is not None:
                totals[mono] = round(totals[mono] + amount, 6)

        ingredient_results.append({
            **item,
            "selected_glycan_id": glycan_id,
            "match_status": mapping.get("match_status"),
            "match_reason": mapping.get("reason"),
            "glycopedia_candidate": mapping.get("selected_candidate"),
            "monosaccharides": mono_intake,
        })

    return {
        "record_id": record_id,
        "basis": "Glycopedia mono values are g per 100g FW; amount_g = value_per_100g * ingredient_grams / 100",
        "ingredients": ingredient_results,
        "daily_monosaccharide_totals_g": totals,
    }


def write_glycopedia_matches_html(match_json_path: Path) -> Path:
    """Write a compact HTML review page for one Glycopedia mapping JSON."""
    with open(match_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for item in data.get("ingredients", []):
        candidate = item.get("glycopedia_candidate") or {}
        rows.append(f"""
            <tr>
                <td>{html_escape(item.get('food_index'))}</td>
                <td>{html_escape(item.get('record_food_description'))}</td>
                <td class="mono">{html_escape(item.get('ingredient_code'))}</td>
                <td>{html_escape(item.get('ingredient_description'))}</td>
                <td>{format_number(item.get('ingredient_grams'), 3)}</td>
                <td>{format_number(item.get('ingredient_weight_fraction'), 4)}</td>
                <td class="mono">{html_escape(item.get('selected_glycan_id'))}</td>
                <td><strong>{html_escape(candidate.get('food_description'))}</strong><br><span>{html_escape(candidate.get('processing'))}</span></td>
                <td>{html_escape(item.get('match_status'))}<br><span>{html_escape(item.get('match_reason'))}</span></td>
            </tr>
""")

    html_path = match_json_path.with_suffix(".html")
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8"/>
    <title>Glycopedia Matches - {html_escape(data.get('record_id'))}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f7f7f7; }}
        .container {{ max-width: 1400px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; vertical-align: top; font-size: 13px; }}
        th {{ background: #eef3f8; }}
        span {{ color: #666; font-size: 12px; }}
        .mono {{ font-family: Menlo, Consolas, monospace; }}
    </style>
</head>
<body>
<div class="container">
    <h1>Glycopedia Matches - {html_escape(data.get('record_id'))}</h1>
    <table>
        <thead>
            <tr>
                <th>Food #</th>
                <th>Record Food</th>
                <th>Ingredient Code</th>
                <th>FNDDS Ingredient</th>
                <th>Ingredient g</th>
                <th>Weight Fraction</th>
                <th>Glycan ID</th>
                <th>Mapped Glycopedia Food</th>
                <th>Status</th>
            </tr>
        </thead>
        <tbody>{''.join(rows)}</tbody>
    </table>
</div>
</body>
</html>
"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    return html_path


def write_glycopedia_results_html(result_json_path: Path) -> Path:
    """Write HTML report for monosaccharide intake results."""
    with open(result_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    totals = data.get("daily_monosaccharide_totals_g", {})
    total_cells = "".join(f"<td>{format_number(totals.get(mono), 4)}</td>" for mono in MONOSACCHARIDE_COLUMNS)

    rows = []
    for item in data.get("ingredients", []):
        mono_cells = "".join(
            f"<td>{format_number(item.get('monosaccharides', {}).get(mono, {}).get('amount_g'), 4)}</td>"
            for mono in MONOSACCHARIDE_COLUMNS
        )
        candidate = item.get("glycopedia_candidate") or {}
        rows.append(f"""
            <tr>
                <td>{html_escape(item.get('food_index'))}</td>
                <td>{html_escape(item.get('record_food_description'))}</td>
                <td>{html_escape(item.get('ingredient_description'))}</td>
                <td>{format_number(item.get('ingredient_grams'), 3)}</td>
                <td>{format_number(item.get('ingredient_weight_fraction'), 4)}</td>
                <td>{html_escape(item.get('selected_glycan_id'))}</td>
                <td><strong>{html_escape(candidate.get('food_description'))}</strong></td>
                <td>{html_escape(item.get('match_status'))}</td>
                {mono_cells}
            </tr>
""")

    mono_headers = "".join(f"<th>{mono}</th>" for mono in MONOSACCHARIDE_COLUMNS)
    html_path = result_json_path.with_suffix(".html")
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8"/>
    <title>Glycopedia Monosaccharides - {html_escape(data.get('record_id'))}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f7f7f7; }}
        .container {{ max-width: 1600px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 16px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; vertical-align: top; font-size: 13px; }}
        th {{ background: #eef3f8; }}
        .scroll {{ overflow-x: auto; }}
    </style>
</head>
<body>
<div class="container">
    <h1>Glycopedia Monosaccharides - {html_escape(data.get('record_id'))}</h1>
    <h2>Daily Total, g</h2>
    <div class="scroll">
        <table>
            <thead><tr>{mono_headers}</tr></thead>
            <tbody><tr>{total_cells}</tr></tbody>
        </table>
    </div>
    <h2>Ingredient Contributions</h2>
    <div class="scroll">
        <table>
            <thead>
                <tr>
                    <th>Food #</th>
                    <th>Record Food</th>
                    <th>FNDDS Ingredient</th>
                    <th>Ingredient g</th>
                    <th>Weight Fraction</th>
                    <th>Glycan ID</th>
                    <th>Mapped Glycopedia Food</th>
                    <th>Status</th>
                    {mono_headers}
                </tr>
            </thead>
            <tbody>{''.join(rows)}</tbody>
        </table>
    </div>
</div>
</body>
</html>
"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    return html_path


def map_records_to_glycopedia(
    candidate_path: Path,
    use_gpt: bool = False,
) -> Tuple[List[Path], List[Path]]:
    """Map existing FNDDS record matches to Glycopedia and calculate mono intakes."""
    print("=" * 80)
    print("Mapping records to Glycopedia")
    print("=" * 80)

    GLYCOPEDIA_MAPPING_DIR.mkdir(parents=True, exist_ok=True)
    GLYCOPEDIA_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    candidate_cache = load_candidate_cache(candidate_path)
    mono_by_id = load_glycopedia_mono_by_id()

    match_json_files = []
    result_json_files = []
    for match_file in sorted(FNDDS_MATCH_DIR.glob("*_matches.json")):
        record_id = match_file.name.replace("_matches.json", "")
        print(f"Processing {record_id}...", end=" ")
        ingredient_items = ingredientize_match_file(match_file)
        ingredient_mapping = build_used_ingredient_mapping(
            ingredient_items,
            candidate_cache,
            use_gpt=use_gpt,
        )
        record_match = {
            "record_id": record_id,
            "use_gpt": use_gpt,
            "ingredients": [
                {
                    **item,
                    **{
                        "selected_glycan_id": ingredient_mapping.get(item["ingredient_code"], {}).get("selected_glycan_id"),
                        "match_status": ingredient_mapping.get(item["ingredient_code"], {}).get("match_status"),
                        "match_reason": ingredient_mapping.get(item["ingredient_code"], {}).get("reason"),
                        "glycopedia_candidate": ingredient_mapping.get(item["ingredient_code"], {}).get("selected_candidate"),
                        "candidates": ingredient_mapping.get(item["ingredient_code"], {}).get("candidates", []),
                    },
                }
                for item in ingredient_items
            ],
        }
        match_out = GLYCOPEDIA_MAPPING_DIR / f"{record_id}_glycopedia_matches.json"
        with open(match_out, "w", encoding="utf-8") as f:
            json.dump(record_match, f, indent=2, ensure_ascii=False)
        write_glycopedia_matches_html(match_out)
        match_json_files.append(match_out)

        mono_result = calculate_record_monosaccharides(
            record_id,
            ingredient_items,
            ingredient_mapping,
            mono_by_id,
        )
        result_out = GLYCOPEDIA_RESULTS_DIR / f"{record_id}_glycopedia_monosaccharides.json"
        with open(result_out, "w", encoding="utf-8") as f:
            json.dump(mono_result, f, indent=2, ensure_ascii=False)
        write_glycopedia_results_html(result_out)
        result_json_files.append(result_out)
        print(f"{len(ingredient_items)} ingredient rows")

    print(f"wrote {len(match_json_files)} mapping JSON files")
    print(f"wrote {len(result_json_files)} result JSON files")
    return match_json_files, result_json_files


def main() -> None:
    """Run Glycopedia mapping steps."""
    parser = argparse.ArgumentParser(description="Map FNDDS record ingredients to Glycopedia monosaccharides.")
    parser.add_argument("--skip-db", action="store_true", help="Do not rebuild db/glycopedia/glycopedia.db")
    parser.add_argument("--skip-embeddings", action="store_true", help="Do not rebuild Glycopedia embeddings")
    parser.add_argument("--skip-candidates", action="store_true", help="Do not rebuild FNDDS ingredient candidates")
    parser.add_argument("--skip-records", action="store_true", help="Do not map records or calculate mono results")
    parser.add_argument("--use-gpt", action="store_true", help="Use GPT to select/reject Glycopedia matches for used ingredients")
    args = parser.parse_args()

    if not args.skip_db:
        build_glycopedia_db()
    elif not GLYCOPEDIA_DB_PATH.exists():
        raise FileNotFoundError(f"{GLYCOPEDIA_DB_PATH} does not exist; rerun without --skip-db")

    if not args.skip_embeddings:
        build_glycopedia_embeddings()
    elif not GLYCOPEDIA_EMBEDDINGS_PATH.exists() and not args.skip_candidates:
        raise FileNotFoundError(
            f"{GLYCOPEDIA_EMBEDDINGS_PATH} does not exist; rerun without --skip-embeddings"
        )

    candidate_path = GLYCOPEDIA_MAPPING_DIR / "fndds_ingredient_glycopedia_candidates.json"
    if not args.skip_candidates:
        candidate_path = build_fndds_ingredient_glycopedia_candidates()
    elif not candidate_path.exists() and not args.skip_records:
        raise FileNotFoundError(f"{candidate_path} does not exist; rerun without --skip-candidates")

    if not args.skip_records:
        map_records_to_glycopedia(candidate_path, use_gpt=args.use_gpt)


if __name__ == "__main__":
    main()
