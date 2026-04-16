#!/usr/bin/env python3
"""
Polyphenol mapping pipeline.

This script is intentionally separate from FNDDS and Glycopedia pipelines.

Conservative v1 workflow:
1. Build a Polyphenol SQLite database from Phenol-Explorer exports.
2. Build a stable Polyphenol food index and embeddings.
3. Build FNDDS ingredient -> Polyphenol food candidates for ingredients used
   in current FNDDS record matches.
4. Optionally use GPT to select or reject Polyphenol food matches.
5. Calculate individual polyphenol intake with method-selection provenance.

V1 intentionally does not do scoped imputation, component expansion, weighted
composites, or cooked/raw yield adjustment.
"""

import argparse
import hashlib
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

POLYPHENOL_DIR = Path("db/polyphenol")
POLYPHENOL_COMPOSITION_XLSX = POLYPHENOL_DIR / "composition-data.xlsx"
POLYPHENOL_COMPOUNDS_CSV = POLYPHENOL_DIR / "compounds-structures.csv"
POLYPHENOL_METABOLITES_CSV = POLYPHENOL_DIR / "metabolites-structures.csv"
POLYPHENOL_DB_PATH = POLYPHENOL_DIR / "polyphenol.db"
POLYPHENOL_EMBEDDINGS_PATH = POLYPHENOL_DIR / "polyphenol_embeddings.npz"

POLYPHENOL_MAPPING_DIR = Path("record/PolyphenolMapping")
POLYPHENOL_RESULTS_DIR = Path("results/polyphenol")

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K_CANDIDATES = 10
SOURCE_VERSION = "Phenol-Explorer local export"
NORMALIZE_VERSION = "v1.0"
WATER_NUTRIENT_CODE = "255"

ANIMAL_OR_LOW_POLYPHENOL_PATTERNS = [
    r"\bwater\b",
    r"\bsalt\b",
    r"\bsugar[s]?\b",
    r"\bsyrup\b",
    r"\bmilk\b",
    r"\bcream\b",
    r"\bbutter\b",
    r"\bcheese\b",
    r"\byogurt\b",
    r"\begg[s]?\b",
    r"\bbeef\b",
    r"\bpork\b",
    r"\bturkey\b",
    r"\bchicken\b",
    r"\bfish\b",
    r"\bsalmon\b",
    r"\bshrimp\b",
    r"\bmayonnaise\b",
]

PLANT_POLYPHENOL_HINT_PATTERNS = [
    r"\bcoffee\b",
    r"\btea\b",
    r"\bcocoa\b",
    r"\bchocolate\b",
    r"\bpeanut butter\b",
    r"\bsoy milk\b",
    r"\bsoy\b",
    r"\bbean[s]?\b",
    r"\blentil[s]?\b",
    r"\bpea[s]?\b",
    r"\bnut[s]?\b",
    r"\bseed[s]?\b",
    r"\bfruit\b",
    r"\bberr",
    r"\btomato",
    r"\blettuce\b",
    r"\bpotato",
    r"\bonion",
    r"\bgarlic",
    r"\bcarrot",
    r"\bcucumber",
    r"\bpepper",
    r"\bartichoke",
    r"\bdate[s]?\b",
    r"\bpear[s]?\b",
    r"\bapple",
    r"\bbread\b",
    r"\bflour\b",
    r"\bwheat\b",
    r"\boat\b",
    r"\brice\b",
    r"\bcorn\b",
    r"\bgraham\b",
    r"\bcracker",
    r"\bcrouton",
    r"\boil\b",
    r"\bherb",
    r"\bspice",
    r"\bmustard\b",
    r"\bsalsa\b",
]


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
    """Normalize text for matching."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    text = str(text)
    text = re.sub(r"\bNFS\b", "not further specified", text)
    text = re.sub(r"\bNS\b", "not specified", text)
    text = text.replace("+", " and ")
    text = text.lower()
    text = re.sub(r"[^\w\s.%/\[\]-]", " ", text)
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
    """Format a number compactly."""
    if isinstance(value, (int, float)) and not pd.isna(value):
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


def stable_polyphenol_food_id(food_group: Any, food_sub_group: Any, food: Any) -> str:
    """Create a stable local food id for a database without native food ids."""
    raw_key = "|".join([
        normalize_text(food_group),
        normalize_text(food_sub_group),
        normalize_text(food),
    ])
    digest = hashlib.sha1(raw_key.encode("utf-8")).hexdigest()[:12]
    return f"PEFOOD_{digest}"


def create_polyphenol_index_text(row: pd.Series) -> str:
    """Create a concise food index text."""
    return "; ".join(
        str(row[col])
        for col in ["food", "food_sub_group", "food_group"]
        if row.get(col) is not None and not pd.isna(row.get(col)) and str(row.get(col)).strip()
    )


def build_polyphenol_db(db_path: Path = POLYPHENOL_DB_PATH) -> Path:
    """Build Polyphenol SQLite database from local exports."""
    print("=" * 80)
    print("Building Polyphenol SQLite database")
    print("=" * 80)

    composition = pd.read_excel(POLYPHENOL_COMPOSITION_XLSX)
    composition = composition.rename(columns={"N": "capital_N"})
    composition["polyphenol_food_id"] = composition.apply(
        lambda row: stable_polyphenol_food_id(row["food_group"], row["food_sub_group"], row["food"]),
        axis=1,
    )
    composition["food_key"] = composition.apply(
        lambda row: "|".join([
            normalize_text(row["food_group"]),
            normalize_text(row["food_sub_group"]),
            normalize_text(row["food"]),
        ]),
        axis=1,
    )

    foods = (
        composition[["polyphenol_food_id", "food_group", "food_sub_group", "food", "food_key"]]
        .drop_duplicates()
        .sort_values(["food_group", "food_sub_group", "food"])
        .reset_index(drop=True)
    )
    foods["normalized_food"] = foods["food"].map(normalize_text)
    foods["index_text"] = foods.apply(create_polyphenol_index_text, axis=1)
    foods["normalized_index_text"] = foods["index_text"].map(normalize_text)
    foods["source_version"] = SOURCE_VERSION
    foods["normalize_version"] = NORMALIZE_VERSION

    composition["source_version"] = SOURCE_VERSION
    composition["normalize_version"] = NORMALIZE_VERSION

    compounds = pd.read_csv(POLYPHENOL_COMPOUNDS_CSV)
    metabolites = pd.read_csv(POLYPHENOL_METABOLITES_CSV)

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        foods.to_sql("polyphenol_foods", conn, if_exists="replace", index=False)
        composition.to_sql("polyphenol_composition", conn, if_exists="replace", index=False)
        compounds.to_sql("polyphenol_compounds", conn, if_exists="replace", index=False)
        metabolites.to_sql("polyphenol_metabolites", conn, if_exists="replace", index=False)
        cur = conn.cursor()
        cur.execute("CREATE INDEX IF NOT EXISTS idx_polyphenol_foods_id ON polyphenol_foods(polyphenol_food_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_polyphenol_foods_key ON polyphenol_foods(food_key)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_polyphenol_composition_food ON polyphenol_composition(polyphenol_food_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_polyphenol_composition_compound ON polyphenol_composition(compound)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_polyphenol_compounds_name ON polyphenol_compounds(name)")
        conn.commit()
    finally:
        conn.close()

    print(f"composition rows: {len(composition)}")
    print(f"unique foods: {len(foods)}")
    print(f"compound structures: {len(compounds)}")
    print(f"metabolite structures: {len(metabolites)}")
    print(f"wrote {db_path}")
    return db_path


def build_polyphenol_embeddings(
    db_path: Path = POLYPHENOL_DB_PATH,
    embeddings_path: Path = POLYPHENOL_EMBEDDINGS_PATH,
    model_name: str = EMBEDDING_MODEL_NAME,
) -> Path:
    """Build Polyphenol food embeddings."""
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        raise RuntimeError("sentence-transformers is required for Polyphenol embeddings")

    print("=" * 80)
    print("Building Polyphenol food embeddings")
    print("=" * 80)
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            """
            SELECT polyphenol_food_id, normalized_index_text
            FROM polyphenol_foods
            ORDER BY polyphenol_food_id
            """
        ).fetchall()
    finally:
        conn.close()

    food_ids = np.array([row[0] for row in rows], dtype=object)
    texts = [row[1] or " " for row in rows]
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(embeddings_path, polyphenol_food_ids=food_ids, embeddings=embeddings)
    print(f"wrote {embeddings_path}")
    return embeddings_path


def load_polyphenol_embeddings(
    embeddings_path: Path = POLYPHENOL_EMBEDDINGS_PATH,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load Polyphenol food embeddings as food ids and matrix."""
    data = np.load(embeddings_path, allow_pickle=True)
    return data["polyphenol_food_ids"], np.asarray(data["embeddings"], dtype=np.float32)


def get_polyphenol_food_metadata_by_id(db_path: Path = POLYPHENOL_DB_PATH) -> Dict[str, Dict[str, Any]]:
    """Load Polyphenol food metadata keyed by local food id."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT f.*,
                   GROUP_CONCAT(DISTINCT c.units) AS available_units,
                   GROUP_CONCAT(DISTINCT c.experimental_method_group) AS available_method_groups,
                   GROUP_CONCAT(DISTINCT c.compound_group) AS available_compound_groups
            FROM polyphenol_foods f
            LEFT JOIN polyphenol_composition c
              ON f.polyphenol_food_id = c.polyphenol_food_id
            GROUP BY f.polyphenol_food_id
            """
        ).fetchall()
        return {row["polyphenol_food_id"]: dict(row) for row in rows}
    finally:
        conn.close()


def cosine_top_k(
    query_embeddings: np.ndarray,
    candidate_embeddings: np.ndarray,
    top_k: int,
) -> List[List[Tuple[int, float]]]:
    """Return candidate matrix indices and cosine scores for each query."""
    q_norm = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
    c_norm = np.linalg.norm(candidate_embeddings, axis=1, keepdims=True)
    sims = (query_embeddings @ candidate_embeddings.T) / (q_norm * c_norm.T + 1e-9)
    out: List[List[Tuple[int, float]]] = []
    for row in sims:
        order = np.argsort(-row)[:top_k]
        out.append([(int(idx), float(row[idx])) for idx in order])
    return out


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
    """Allocate consumed FNDDS food grams across ingredient rows."""
    weights = [safe_float(ingredient.get("Ingredient_weight")) for ingredient in ingredients]
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


def get_used_fndds_ingredients() -> List[Dict[str, str]]:
    """Return unique FNDDS ingredients used by existing FNDDS record matches."""
    unique: Dict[str, str] = {}
    for match_file in sorted(FNDDS_MATCH_DIR.glob("*_matches.json")):
        for item in ingredientize_match_file(match_file):
            code = str(item.get("ingredient_code") or "")
            desc = str(item.get("ingredient_description") or "")
            if code:
                unique.setdefault(code, desc)
    return [
        {
            "ingredient_code": code,
            "ingredient_description": desc,
            "normalized_ingredient_description": normalize_text(desc),
            "preclassified_status": preclassify_fndds_ingredient(desc),
        }
        for code, desc in sorted(unique.items(), key=lambda x: x[0])
    ]


def preclassify_fndds_ingredient(description: str) -> str:
    """Quick conservative preclassification before semantic matching."""
    norm = normalize_text(description)
    for pattern in PLANT_POLYPHENOL_HINT_PATTERNS:
        if re.search(pattern, norm):
            return "candidate_for_polyphenol"
    for pattern in ANIMAL_OR_LOW_POLYPHENOL_PATTERNS:
        if re.search(pattern, norm):
            return "not_applicable_likely_no_polyphenols"
    return "candidate_for_polyphenol"


def build_fndds_ingredient_polyphenol_candidates(
    db_path: Path = POLYPHENOL_DB_PATH,
    embeddings_path: Path = POLYPHENOL_EMBEDDINGS_PATH,
    top_k: int = TOP_K_CANDIDATES,
    model_name: str = EMBEDDING_MODEL_NAME,
) -> Path:
    """Build top-k Polyphenol food candidates for used FNDDS ingredients."""
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        raise RuntimeError("sentence-transformers is required for candidate search")

    print("=" * 80)
    print("Building used FNDDS ingredient -> Polyphenol food candidates")
    print("=" * 80)

    ingredients = get_used_fndds_ingredients()
    food_ids, food_embeddings = load_polyphenol_embeddings(embeddings_path)
    food_meta = get_polyphenol_food_metadata_by_id(db_path)
    model = SentenceTransformer(model_name)
    query_texts = [item["normalized_ingredient_description"] or " " for item in ingredients]
    query_embeddings = model.encode(query_texts, show_progress_bar=True, convert_to_numpy=True)
    query_embeddings = np.asarray(query_embeddings, dtype=np.float32)
    ranked = cosine_top_k(query_embeddings, food_embeddings, top_k)

    candidate_rows = []
    candidate_json: Dict[str, Any] = {
        "source": "Used FNDDS ingredient descriptions vs Polyphenol food index",
        "embedding_model": model_name,
        "top_k": top_k,
        "ingredients": [],
    }
    for ingredient, top_matches in zip(ingredients, ranked):
        candidates = []
        for rank, (idx, score) in enumerate(top_matches, start=1):
            food_id = str(food_ids[idx])
            meta = food_meta.get(food_id, {})
            candidate = {
                "rank": rank,
                "polyphenol_food_id": food_id,
                "food": meta.get("food"),
                "food_group": meta.get("food_group"),
                "food_sub_group": meta.get("food_sub_group"),
                "available_units": meta.get("available_units"),
                "available_method_groups": meta.get("available_method_groups"),
                "available_compound_groups": meta.get("available_compound_groups"),
                "similarity_score": score,
            }
            candidates.append(candidate)
            candidate_rows.append({
                "ingredient_code": ingredient["ingredient_code"],
                "ingredient_description": ingredient["ingredient_description"],
                "rank": rank,
                "polyphenol_food_id": food_id,
                "polyphenol_food": meta.get("food"),
                "similarity_score": score,
            })
        candidate_json["ingredients"].append({
            **ingredient,
            "candidates": candidates,
        })

    conn = sqlite3.connect(str(db_path))
    try:
        pd.DataFrame(candidate_rows).to_sql(
            "fndds_ingredient_polyphenol_candidates",
            conn,
            if_exists="replace",
            index=False,
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_fndds_polyphenol_candidates_ingredient ON fndds_ingredient_polyphenol_candidates(ingredient_code)"
        )
        conn.commit()
    finally:
        conn.close()

    POLYPHENOL_MAPPING_DIR.mkdir(parents=True, exist_ok=True)
    out_path = POLYPHENOL_MAPPING_DIR / "fndds_ingredient_polyphenol_candidates.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(candidate_json, f, indent=2, ensure_ascii=False)

    print(f"used ingredients: {len(ingredients)}")
    print(f"wrote {out_path}")
    return out_path


def load_candidate_cache(candidate_path: Path) -> Dict[str, Dict[str, Any]]:
    """Load candidate JSON as ingredient_code -> item with candidate list."""
    with open(candidate_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        item["ingredient_code"]: item
        for item in data.get("ingredients", [])
    }


def select_polyphenol_match_with_gpt(
    ingredient_description: str,
    preclassified_status: str,
    candidates: List[Dict[str, Any]],
    client: Optional["OpenAI"],
) -> Dict[str, Any]:
    """Ask GPT to select one Polyphenol food candidate or reject all."""
    if preclassified_status.startswith("not_applicable"):
        return {
            "selected_polyphenol_food_id": None,
            "match_status": "not_applicable",
            "mapping_type": "excluded_non_polyphenol",
            "confidence": "high",
            "reason": "Ingredient is water, animal food, dairy, salt, sugar, or another item expected to contain no meaningful plant polyphenols in v1.",
        }
    if not candidates or client is None:
        return {
            "selected_polyphenol_food_id": None,
            "match_status": "needs_review_embedding_only" if candidates else "no_candidates",
            "mapping_type": None,
            "confidence": "low",
            "reason": "No GPT selection was run; embedding candidates are suggestions only." if candidates else "No Polyphenol food candidates found.",
        }

    candidates_text = "\n".join(
        "- polyphenol_food_id: {polyphenol_food_id} | food: {food} | group: {food_group} | "
        "subgroup: {food_sub_group} | units: {available_units} | methods: {available_method_groups} "
        "| compound groups: {available_compound_groups} | similarity: {similarity_score:.4f}".format(**candidate)
        for candidate in candidates
    )
    prompt = f"""You are mapping an FNDDS ingredient to a Phenol-Explorer / Polyphenol database food for conservative polyphenol intake estimation.

FNDDS ingredient description:
{ingredient_description}

Polyphenol food candidates:
{candidates_text}

V1 rules:
- Select one candidate only if it is a plausible edible-form match.
- Prefer direct food matches already present in the Polyphenol database.
- Do not do component expansion, wheat-flour imputation, scoped imputation, cooked/raw yield adjustment, or weighted composites.
- Reject animal foods, water, salt, plain dairy, pure sugar, and foods with no meaningful plant polyphenols.
- Reject when the form is wrong, such as dry powder vs brewed beverage, raw vs cooked when not acceptable, or fresh vs dried when unclear.

Return ONLY valid JSON:
{{
  "selected_polyphenol_food_id": "PEFOOD_abc123" or null,
  "match_status": "matched" or "not_applicable" or "no_confident_match",
  "mapping_type": "direct_food_match" or "ingredient_direct_match" or "excluded_non_polyphenol" or null,
  "confidence": "high" or "medium" or "low",
  "reason": "short reason"
}}"""
    try:
        response = client.chat.completions.create(
            model="gpt-5.2",
            messages=[
                {"role": "system", "content": "You select or reject conservative Polyphenol food matches. Return only JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_completion_tokens=300,
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
        return {
            "selected_polyphenol_food_id": None,
            "match_status": "needs_review_embedding_only",
            "mapping_type": None,
            "confidence": "low",
            "reason": f"GPT selection failed: {exc}",
        }

    selected = result.get("selected_polyphenol_food_id")
    valid_ids = {candidate["polyphenol_food_id"] for candidate in candidates}
    if selected not in valid_ids:
        selected = None
    return {
        "selected_polyphenol_food_id": selected,
        "match_status": result.get("match_status") or ("matched" if selected else "no_confident_match"),
        "mapping_type": result.get("mapping_type"),
        "confidence": result.get("confidence"),
        "reason": result.get("reason", ""),
    }


def build_used_ingredient_mapping(
    ingredient_items: List[Dict[str, Any]],
    candidate_cache: Dict[str, Dict[str, Any]],
    use_gpt: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """Build selected Polyphenol food mapping for used ingredients."""
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
        cache_item = candidate_cache.get(ingredient_code, {})
        candidates = cache_item.get("candidates", [])
        preclassified_status = cache_item.get("preclassified_status") or preclassify_fndds_ingredient(description)
        selected = select_polyphenol_match_with_gpt(description, preclassified_status, candidates, client)
        selected_candidate = next(
            (
                candidate
                for candidate in candidates
                if candidate.get("polyphenol_food_id") == selected.get("selected_polyphenol_food_id")
            ),
            None,
        )
        mapping[ingredient_code] = {
            "ingredient_code": ingredient_code,
            "ingredient_description": description,
            "preclassified_status": preclassified_status,
            "selected_polyphenol_food_id": selected.get("selected_polyphenol_food_id"),
            "match_status": selected.get("match_status"),
            "mapping_type": selected.get("mapping_type"),
            "confidence": selected.get("confidence"),
            "reason": selected.get("reason"),
            "selected_candidate": selected_candidate,
            "candidates": candidates,
        }
    return mapping


def load_water_percent_by_ingredient_code(fndds_db_path: Path = FNDDS_DB_PATH) -> Dict[str, float]:
    """Load FNDDS water g/100g for ingredient codes."""
    conn = sqlite3.connect(str(fndds_db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT Ingredient_code, Nutrient_value
            FROM ingrednutval
            WHERE Nutrient_code = ?
            """,
            (WATER_NUTRIENT_CODE,),
        ).fetchall()
    finally:
        conn.close()
    return {
        str(row["Ingredient_code"]): float(row["Nutrient_value"])
        for row in rows
        if row["Nutrient_value"] is not None
    }


def should_include_composition_row(row: sqlite3.Row) -> Tuple[bool, bool, str]:
    """
    Return whether a composition row is included in individual or Folin totals.

    The v1 individual total follows the paper's method-selection logic without
    adding scoped imputation.
    """
    method = str(row["experimental_method_group"] or "")
    compound_group = str(row["compound_group"] or "")
    compound_sub_group = str(row["compound_sub_group"] or "")
    compound = str(row["compound"] or "")
    food_group = str(row["food_group"] or "")
    food_sub_group = str(row["food_sub_group"] or "")
    food = str(row["food"] or "")
    text = normalize_text(" ".join([food_group, food_sub_group, food, compound_group, compound_sub_group, compound]))

    if method == "Folin assay":
        return False, True, "Folin assay is reported separately and excluded from individual-compound totals."
    if method == "Normal phase HPLC (proanthocyanidins)":
        return True, False, "Normal phase HPLC is used for proanthocyanidins."
    if method == "Chromatography after hydrolysis":
        if "lignan" in normalize_text(compound_sub_group):
            return True, False, "Chromatography after hydrolysis is used for lignans."
        if "walnut" in text and normalize_text(compound) == "ellagic acid":
            return True, False, "Chromatography after hydrolysis is used for ellagic acid in walnuts."
        if (
            "hydroxycinnamic acids" in normalize_text(compound_sub_group)
            and any(term in text for term in ["cereal", "wheat", "bread", "flour", "rice", "oat", "barley", "maize", "white bean", "olive"])
        ):
            return True, False, "Chromatography after hydrolysis is used for hydroxycinnamic acids in cereals, white beans, and olives."
        return False, False, "Hydrolysis row is outside v1 paper-based inclusion rules."
    if method == "Chromatography":
        return True, False, "Chromatography is used for individual compounds by default."
    return False, False, "Unknown experimental method is excluded in v1."


def load_composition_rows_for_food(
    polyphenol_food_id: str,
    db_path: Path = POLYPHENOL_DB_PATH,
) -> List[Dict[str, Any]]:
    """Load selected composition rows for a Polyphenol food."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT *
            FROM polyphenol_composition
            WHERE polyphenol_food_id = ?
            ORDER BY compound_group, compound_sub_group, compound, experimental_method_group, units
            """,
            (polyphenol_food_id,),
        ).fetchall()
    finally:
        conn.close()

    out = []
    for row in rows:
        include_individual, include_folin, reason = should_include_composition_row(row)
        row_dict = dict(row)
        row_dict["included_in_individual_total"] = include_individual
        row_dict["included_in_folin_total"] = include_folin
        row_dict["method_selection_reason"] = reason
        out.append(row_dict)
    return out


def calculate_polyphenol_amount(
    composition_row: Dict[str, Any],
    ingredient_item: Dict[str, Any],
    water_percent_by_ingredient_code: Dict[str, float],
) -> Dict[str, Any]:
    """Calculate intake amount for one composition row and one ingredient."""
    ingredient_grams = safe_float(ingredient_item.get("ingredient_grams"))
    mean = safe_float(composition_row.get("mean"))
    units = str(composition_row.get("units") or "")
    ingredient_code = str(ingredient_item.get("ingredient_code"))
    if ingredient_grams is None or mean is None:
        return {"amount_mg": None, "calculation_status": "missing_grams_or_mean"}

    if units == "mg/100 g fresh weight":
        return {
            "amount_mg": round(ingredient_grams * mean / 100.0, 6),
            "calculation_status": "calculated",
            "calculation_basis": "ingredient_grams_as_consumed * mg_per_100g_fresh_weight / 100",
        }
    if units in {"mg/100 ml", "mg/100 mL"}:
        return {
            "amount_mg": round(ingredient_grams * mean / 100.0, 6),
            "calculation_status": "calculated_with_density_assumption",
            "calculation_basis": "ingredient_grams treated as ml using density_1g_per_ml",
            "unit_assumption": "density_1g_per_ml",
        }
    if units == "mg/100 g dry weight":
        water_percent = water_percent_by_ingredient_code.get(ingredient_code)
        if water_percent is None:
            return {
                "amount_mg": None,
                "calculation_status": "needs_moisture_conversion",
                "calculation_basis": "dry-weight composition requires FNDDS water nutrient 255",
            }
        dry_grams = ingredient_grams * (1.0 - water_percent / 100.0)
        return {
            "amount_mg": round(dry_grams * mean / 100.0, 6),
            "calculation_status": "calculated_dry_weight_from_fndds_water",
            "calculation_basis": "ingredient_grams * (1 - water_percent / 100) * mg_per_100g_dry_weight / 100",
            "water_percent": water_percent,
            "dry_grams": round(dry_grams, 6),
        }
    return {"amount_mg": None, "calculation_status": f"unsupported_units:{units}"}


def calculate_record_polyphenols(
    record_id: str,
    ingredient_items: List[Dict[str, Any]],
    ingredient_mapping: Dict[str, Dict[str, Any]],
    water_percent_by_ingredient_code: Dict[str, float],
) -> Dict[str, Any]:
    """Calculate Polyphenol intake for one record."""
    ingredient_results = []
    compound_totals: Dict[str, float] = {}
    compound_group_totals: Dict[str, float] = {}
    compound_sub_group_totals: Dict[str, float] = {}
    food_source_totals: Dict[str, float] = {}
    folin_totals: Dict[str, float] = {}

    for item in ingredient_items:
        mapping = ingredient_mapping.get(item["ingredient_code"], {})
        food_id = mapping.get("selected_polyphenol_food_id")
        composition_rows = load_composition_rows_for_food(food_id) if food_id else []
        compound_rows = []
        for comp_row in composition_rows:
            calc = calculate_polyphenol_amount(comp_row, item, water_percent_by_ingredient_code)
            amount = calc.get("amount_mg")
            row_out = {
                "compound": comp_row.get("compound"),
                "compound_group": comp_row.get("compound_group"),
                "compound_sub_group": comp_row.get("compound_sub_group"),
                "experimental_method_group": comp_row.get("experimental_method_group"),
                "units": comp_row.get("units"),
                "mean": comp_row.get("mean"),
                "publication_ids": comp_row.get("publication_ids"),
                "pubmed_ids": comp_row.get("pubmed_ids"),
                "included_in_individual_total": comp_row.get("included_in_individual_total"),
                "included_in_folin_total": comp_row.get("included_in_folin_total"),
                "method_selection_reason": comp_row.get("method_selection_reason"),
                **calc,
            }
            compound_rows.append(row_out)
            if amount is None:
                continue
            if comp_row.get("included_in_individual_total"):
                compound = str(comp_row.get("compound"))
                compound_group = str(comp_row.get("compound_group"))
                compound_sub_group = str(comp_row.get("compound_sub_group"))
                food_name = (mapping.get("selected_candidate") or {}).get("food") or food_id
                compound_totals[compound] = round(compound_totals.get(compound, 0.0) + amount, 6)
                compound_group_totals[compound_group] = round(compound_group_totals.get(compound_group, 0.0) + amount, 6)
                compound_sub_group_totals[compound_sub_group] = round(compound_sub_group_totals.get(compound_sub_group, 0.0) + amount, 6)
                food_source_totals[str(food_name)] = round(food_source_totals.get(str(food_name), 0.0) + amount, 6)
            if comp_row.get("included_in_folin_total"):
                food_name = (mapping.get("selected_candidate") or {}).get("food") or food_id
                folin_totals[str(food_name)] = round(folin_totals.get(str(food_name), 0.0) + amount, 6)

        ingredient_results.append({
            **item,
            "preclassified_status": mapping.get("preclassified_status"),
            "selected_polyphenol_food_id": food_id,
            "match_status": mapping.get("match_status"),
            "mapping_type": mapping.get("mapping_type"),
            "match_confidence": mapping.get("confidence"),
            "match_reason": mapping.get("reason"),
            "polyphenol_candidate": mapping.get("selected_candidate"),
            "polyphenol_compounds": compound_rows,
        })

    total_individual = round(sum(compound_totals.values()), 6)
    return {
        "record_id": record_id,
        "basis": "Conservative v1: no imputation, no component expansion, no yield adjustment. Individual totals use paper-inspired method selection; Folin assay is separate.",
        "ingredients": ingredient_results,
        "daily_total_individual_polyphenols_mg": total_individual,
        "daily_compound_totals_mg": dict(sorted(compound_totals.items(), key=lambda x: -x[1])),
        "daily_compound_group_totals_mg": dict(sorted(compound_group_totals.items(), key=lambda x: -x[1])),
        "daily_compound_sub_group_totals_mg": dict(sorted(compound_sub_group_totals.items(), key=lambda x: -x[1])),
        "daily_food_source_totals_mg": dict(sorted(food_source_totals.items(), key=lambda x: -x[1])),
        "folin_total_polyphenols_by_food_source_mg": dict(sorted(folin_totals.items(), key=lambda x: -x[1])),
    }


def write_polyphenol_matches_html(match_json_path: Path) -> Path:
    """Write a compact HTML review page for one Polyphenol mapping JSON."""
    with open(match_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for item in data.get("ingredients", []):
        candidate = item.get("polyphenol_candidate") or {}
        top_candidates = item.get("candidates", [])[:3]
        suggestions = "<br>".join(
            f"{html_escape(c.get('rank'))}. {html_escape(c.get('food'))} ({format_number(c.get('similarity_score'), 3)})"
            for c in top_candidates
        )
        rows.append(f"""
            <tr>
                <td>{html_escape(item.get('food_index'))}</td>
                <td>{html_escape(item.get('record_food_description'))}</td>
                <td class="mono">{html_escape(item.get('ingredient_code'))}</td>
                <td>{html_escape(item.get('ingredient_description'))}</td>
                <td>{format_number(item.get('ingredient_grams'), 3)}</td>
                <td>{html_escape(item.get('preclassified_status'))}</td>
                <td class="mono">{html_escape(item.get('selected_polyphenol_food_id'))}</td>
                <td><strong>{html_escape(candidate.get('food'))}</strong><br><span>{html_escape(candidate.get('food_group'))} / {html_escape(candidate.get('food_sub_group'))}</span></td>
                <td>{html_escape(item.get('match_status'))}<br><span>{html_escape(item.get('match_reason'))}</span></td>
                <td>{suggestions}</td>
            </tr>
""")

    html_path = match_json_path.with_suffix(".html")
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8"/>
    <title>Polyphenol Matches - {html_escape(data.get('record_id'))}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f7f7f7; }}
        .container {{ max-width: 1500px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; vertical-align: top; font-size: 13px; }}
        th {{ background: #eef3f8; }}
        span {{ color: #666; font-size: 12px; }}
        .mono {{ font-family: Menlo, Consolas, monospace; }}
    </style>
</head>
<body>
<div class="container">
    <h1>Polyphenol Matches - {html_escape(data.get('record_id'))}</h1>
    <p>Conservative v1: embedding candidates are suggestions unless GPT selection was used.</p>
    <table>
        <thead>
            <tr>
                <th>Food #</th>
                <th>Record Food</th>
                <th>Ingredient Code</th>
                <th>FNDDS Ingredient</th>
                <th>Ingredient g</th>
                <th>Preclass</th>
                <th>Polyphenol Food ID</th>
                <th>Mapped Food</th>
                <th>Status</th>
                <th>Top Suggestions</th>
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


def write_polyphenol_results_html(result_json_path: Path) -> Path:
    """Write HTML report for Polyphenol intake results."""
    with open(result_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    group_rows = "".join(
        f"<tr><td>{html_escape(k)}</td><td>{format_number(v, 4)}</td></tr>"
        for k, v in data.get("daily_compound_group_totals_mg", {}).items()
    )
    source_rows = "".join(
        f"<tr><td>{html_escape(k)}</td><td>{format_number(v, 4)}</td></tr>"
        for k, v in data.get("daily_food_source_totals_mg", {}).items()
    )
    compound_rows = "".join(
        f"<tr><td>{html_escape(k)}</td><td>{format_number(v, 4)}</td></tr>"
        for k, v in list(data.get("daily_compound_totals_mg", {}).items())[:100]
    )

    ingredient_rows = []
    for item in data.get("ingredients", []):
        candidate = item.get("polyphenol_candidate") or {}
        included_amount = sum(
            safe_float(row.get("amount_mg")) or 0.0
            for row in item.get("polyphenol_compounds", [])
            if row.get("included_in_individual_total")
        )
        ingredient_rows.append(f"""
            <tr>
                <td>{html_escape(item.get('food_index'))}</td>
                <td>{html_escape(item.get('record_food_description'))}</td>
                <td>{html_escape(item.get('ingredient_description'))}</td>
                <td>{format_number(item.get('ingredient_grams'), 3)}</td>
                <td><strong>{html_escape(candidate.get('food'))}</strong></td>
                <td>{html_escape(item.get('match_status'))}</td>
                <td>{format_number(included_amount, 4)}</td>
            </tr>
""")

    html_path = result_json_path.with_suffix(".html")
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8"/>
    <title>Polyphenol Intakes - {html_escape(data.get('record_id'))}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f7f7f7; }}
        .container {{ max-width: 1500px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 20px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 12px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; vertical-align: top; font-size: 13px; }}
        th {{ background: #eef3f8; }}
        .total {{ font-size: 24px; font-weight: bold; }}
        .scroll {{ overflow-x: auto; }}
    </style>
</head>
<body>
<div class="container">
    <h1>Polyphenol Intakes - {html_escape(data.get('record_id'))}</h1>
    <p class="total">Daily individual polyphenols: {format_number(data.get('daily_total_individual_polyphenols_mg'), 3)} mg</p>
    <p>{html_escape(data.get('basis'))}</p>
    <div class="grid">
        <section>
            <h2>By Compound Group</h2>
            <table><thead><tr><th>Group</th><th>mg</th></tr></thead><tbody>{group_rows}</tbody></table>
        </section>
        <section>
            <h2>By Food Source</h2>
            <table><thead><tr><th>Food</th><th>mg</th></tr></thead><tbody>{source_rows}</tbody></table>
        </section>
    </div>
    <h2>Ingredient Contributions</h2>
    <div class="scroll">
        <table>
            <thead><tr><th>Food #</th><th>Record Food</th><th>FNDDS Ingredient</th><th>Ingredient g</th><th>Mapped Food</th><th>Status</th><th>Individual mg</th></tr></thead>
            <tbody>{''.join(ingredient_rows)}</tbody>
        </table>
    </div>
    <h2>Top Compounds</h2>
    <table><thead><tr><th>Compound</th><th>mg</th></tr></thead><tbody>{compound_rows}</tbody></table>
</div>
</body>
</html>
"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    return html_path


def map_records_to_polyphenol(
    candidate_path: Path,
    use_gpt: bool = False,
) -> Tuple[List[Path], List[Path]]:
    """Map existing FNDDS record matches to Polyphenol foods and calculate intakes."""
    print("=" * 80)
    print("Mapping records to Polyphenol database")
    print("=" * 80)

    POLYPHENOL_MAPPING_DIR.mkdir(parents=True, exist_ok=True)
    POLYPHENOL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    candidate_cache = load_candidate_cache(candidate_path)
    water_percent_by_ingredient_code = load_water_percent_by_ingredient_code()

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
                    "preclassified_status": ingredient_mapping.get(item["ingredient_code"], {}).get("preclassified_status"),
                    "selected_polyphenol_food_id": ingredient_mapping.get(item["ingredient_code"], {}).get("selected_polyphenol_food_id"),
                    "match_status": ingredient_mapping.get(item["ingredient_code"], {}).get("match_status"),
                    "mapping_type": ingredient_mapping.get(item["ingredient_code"], {}).get("mapping_type"),
                    "match_confidence": ingredient_mapping.get(item["ingredient_code"], {}).get("confidence"),
                    "match_reason": ingredient_mapping.get(item["ingredient_code"], {}).get("reason"),
                    "polyphenol_candidate": ingredient_mapping.get(item["ingredient_code"], {}).get("selected_candidate"),
                    "candidates": ingredient_mapping.get(item["ingredient_code"], {}).get("candidates", []),
                }
                for item in ingredient_items
            ],
        }
        match_out = POLYPHENOL_MAPPING_DIR / f"{record_id}_polyphenol_matches.json"
        with open(match_out, "w", encoding="utf-8") as f:
            json.dump(record_match, f, indent=2, ensure_ascii=False)
        write_polyphenol_matches_html(match_out)
        match_json_files.append(match_out)

        result = calculate_record_polyphenols(
            record_id,
            ingredient_items,
            ingredient_mapping,
            water_percent_by_ingredient_code,
        )
        result_out = POLYPHENOL_RESULTS_DIR / f"{record_id}_polyphenol_intakes.json"
        with open(result_out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        write_polyphenol_results_html(result_out)
        result_json_files.append(result_out)
        print(f"{len(ingredient_items)} ingredient rows")

    print(f"wrote {len(match_json_files)} mapping JSON files")
    print(f"wrote {len(result_json_files)} result JSON files")
    return match_json_files, result_json_files


def main() -> None:
    """Run Polyphenol mapping steps."""
    parser = argparse.ArgumentParser(description="Map FNDDS record ingredients to Polyphenol composition values.")
    parser.add_argument("--skip-db", action="store_true", help="Do not rebuild db/polyphenol/polyphenol.db")
    parser.add_argument("--skip-embeddings", action="store_true", help="Do not rebuild Polyphenol food embeddings")
    parser.add_argument("--skip-candidates", action="store_true", help="Do not rebuild FNDDS ingredient candidates")
    parser.add_argument("--skip-records", action="store_true", help="Do not map records or calculate results")
    parser.add_argument("--use-gpt", action="store_true", help="Use GPT to select/reject Polyphenol matches")
    args = parser.parse_args()

    if not args.skip_db:
        build_polyphenol_db()
    elif not POLYPHENOL_DB_PATH.exists():
        raise FileNotFoundError(f"{POLYPHENOL_DB_PATH} does not exist; rerun without --skip-db")

    if not args.skip_embeddings:
        build_polyphenol_embeddings()
    elif not POLYPHENOL_EMBEDDINGS_PATH.exists() and not args.skip_candidates:
        raise FileNotFoundError(
            f"{POLYPHENOL_EMBEDDINGS_PATH} does not exist; rerun without --skip-embeddings"
        )

    candidate_path = POLYPHENOL_MAPPING_DIR / "fndds_ingredient_polyphenol_candidates.json"
    if not args.skip_candidates:
        candidate_path = build_fndds_ingredient_polyphenol_candidates()
    elif not candidate_path.exists() and not args.skip_records:
        raise FileNotFoundError(f"{candidate_path} does not exist; rerun without --skip-candidates")

    if not args.skip_records:
        map_records_to_polyphenol(candidate_path, use_gpt=args.use_gpt)


if __name__ == "__main__":
    main()
