#!/usr/bin/env python3
"""
FNDDS Record Mapping Pipeline

This script performs the following steps:
1. Normalizes FNDDS food descriptions and creates fndds_food_index table
2. Precomputes FNDDS candidate embeddings (sentence-transformers/all-MiniLM-L6-v2) and saves as numpy file
3. Parses reviewed HTML records and generates JSON objects
4. Normalizes food descriptions using GPT-5.2 to extract structured information
5. Matches records to FNDDS by embedding similarity, returns top 10; GPT selects one final candidate per food by meaning
6. Maps the original amount/unit to one portion code for the GPT-selected FNDDS food, then converts amount to grams
7. Generates visualization HTML: GPT selection shown first, then top 10 with similarity scores and portion/gram conversion
8. Calculates FNDDS nutrient amounts per food and daily nutrient totals into results/fndds
9. Generates readable HTML reports for FNDDS nutrient results

All code follows standard Python practices with type hints and docstrings.
"""

import sqlite3
import json
import re
import os
import html
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from bs4 import BeautifulSoup

import numpy as np

# Optional: sentence-transformers for embedding rerank
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Try to import OpenAI, provide helpful error if not available
try:
    from openai import OpenAI
    from dotenv import load_dotenv
    load_dotenv()
    OPENAI_AVAILABLE = True
except ImportError:
    print("Warning: OpenAI package not available. GPT normalization will be skipped.")
    print("Install with: pip install openai python-dotenv")
    OPENAI_AVAILABLE = False


# Constants
FNDDS_DB_PATH = Path("db/fndds/fndds_2021_2023.db")
REVIEW_DIR = Path("record/Review")
OUTPUT_DIR = Path("record/FNDDSMapping")
FNDDS_RESULTS_DIR = Path("results/fndds")
SOURCE_VERSION = "FNDDS 2021-2023"
NORMALIZE_VERSION = "v1.0"
EMBEDDING_RERANK_TOP_K = 10
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# Precomputed FNDDS embeddings path (food_code -> embedding); stored next to DB
FNDDS_EMBEDDINGS_PATH = Path("db/fndds/fndds_embeddings.npz")
# Legacy name kept for any code that referenced it
TOP_K_MATCHES = EMBEDDING_RERANK_TOP_K


MASS_UNIT_TO_GRAMS = {
    "g": 1.0,
    "gram": 1.0,
    "grams": 1.0,
    "kg": 1000.0,
    "kilogram": 1000.0,
    "kilograms": 1000.0,
    "oz": 28.349523125,
    "ounce": 28.349523125,
    "ounces": 28.349523125,
    "lb": 453.59237,
    "lbs": 453.59237,
    "pound": 453.59237,
    "pounds": 453.59237,
}

UNIT_SYNONYMS = {
    "t": "teaspoon",
    "tsp": "teaspoon",
    "teaspoons": "teaspoon",
    "teaspoon": "teaspoon",
    "T": "tablespoon",
    "tbsp": "tablespoon",
    "tablespoons": "tablespoon",
    "tablespoon": "tablespoon",
    "c": "cup",
    "cups": "cup",
    "cup": "cup",
    "fl oz": "fl oz",
    "fluid ounce": "fl oz",
    "fluid ounces": "fl oz",
    "floz": "fl oz",
    "ml": "ml",
    "milliliter": "ml",
    "milliliters": "ml",
    "l": "liter",
    "liter": "liter",
    "liters": "liter",
    "slice": "slice",
    "slices": "slice",
    "piece": "piece",
    "pieces": "piece",
    "packet": "packet",
    "packets": "packet",
    "container": "container",
    "containers": "container",
    "date": "date",
    "dates": "date",
}

PORTION_UNIT_PATTERNS = [
    ("tablespoon", r"\b(?:tbsp|tablespoons?|T)\b"),
    ("teaspoon", r"\b(?:tsp|teaspoons?|t)\b"),
    ("fl oz", r"\b(?:fl\s*oz|fluid\s+ounces?)\b"),
    ("oz", r"(?<!fl\s)\b(?:oz|ounces?)\b"),
    ("cup", r"\bcups?\b"),
    ("ml", r"\b(?:ml|milliliters?)\b"),
    ("liter", r"\b(?:l|liters?)\b"),
    ("slice", r"\bslices?\b"),
    ("piece", r"\bpieces?\b"),
    ("packet", r"\bpackets?\b"),
    ("container", r"\bcontainers?\b"),
    ("date", r"\bdates?\b"),
]

VOLUME_UNIT_IN_TABLESPOONS = {
    "teaspoon": 1.0 / 3.0,
    "tablespoon": 1.0,
    "fl oz": 2.0,
    "cup": 16.0,
}


def create_openai_client(api_key: Optional[str] = None) -> OpenAI:
    """Create an OpenAI client, honoring an optional regional base URL."""
    resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not resolved_api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        return OpenAI(api_key=resolved_api_key, base_url=base_url)
    return OpenAI(api_key=resolved_api_key)


def is_incorrect_hostname_error(exc: Exception) -> bool:
    """Return True when the API asks for a different regional hostname."""
    message = str(exc)
    return "incorrect regional hostname" in message.lower() or "incorrect_hostname" in message.lower()


@dataclass
class FoodIndexEntry:
    """Represents an entry in the fndds_food_index table."""
    food_code: str
    main_food_description: str
    normalized_food_description: str
    source_version: str
    normalize_version: str


@dataclass
class RecordFoodItem:
    """Represents a food item from a reviewed record."""
    time: str
    place: str
    food_description: str
    amount: str
    unit: str


@dataclass
class NormalizedFoodItem:
    """Represents a normalized food item with structured information."""
    time: str
    place: str
    food_description: str
    normalized_description: str
    core_food: str
    modifiers: List[str]
    process_method: Optional[str]
    amount: str
    unit: str


@dataclass
class RecordData:
    """Represents a complete record with metadata and food items."""
    record_id: str
    subject_id: str
    day_of_week: str
    diet_type: str
    date_of_record: str
    reviewer_id: str
    foods: List[RecordFoodItem]


@dataclass
class MatchResult:
    """Represents a match result (food_code, description, score)."""
    food_code: str
    main_food_description: str
    match_score: float


@dataclass
class FoodMatch:
    """Represents matches for a single food item."""
    food_description: str
    normalized_description: str
    amount: str
    unit: str
    matches: List[MatchResult]


@dataclass
class RecordMatch:
    """Represents all matches for a complete record."""
    record_id: str
    metadata: Dict[str, str]
    food_matches: List[FoodMatch]


def normalize_text(text: str) -> str:
    """
    Normalize text according to v1.0 rules:
    - Capitalized "NFS" (whole word) -> "not further specified"; "NS" (whole word) -> "not specified"
    - Replace "+" with " and "
    - Convert to lowercase
    - Keep percent (%), decimals (digit.digit), and fractions (digit/digit); remove other punctuation
    - Single space between words
    
    Args:
        text: Input text to normalize
        
    Returns:
        Normalized text string
    """
    if not text:
        return ""
    
    # Replace capitalized NFS/NS before lowercasing (NFS first so NS in NFS is not replaced)
    text = re.sub(r"\bNFS\b", "not further specified", text)
    text = re.sub(r"\bNS\b", "not specified", text)
    
    # Replace + with " and "
    text = text.replace("+", " and ")
    
    # Convert to lowercase
    text = text.lower()
    
    # Keep alphanumeric, space, %, ., / (we will strip . and / when not used as decimal/fraction)
    text = re.sub(r"[^\w\s.%/]", " ", text)
    # Keep only decimal point (.) between digits, not period; keep only fraction (/) between digits
    text = re.sub(r"(?<!\d)\.(?!\d)", " ", text)
    text = re.sub(r"(?<!\d)/(?!\d)", " ", text)
    
    # Replace multiple whitespace with single space
    text = re.sub(r"\s+", " ", text)
    
    # Strip leading/trailing spaces
    return text.strip()


def normalize_unit(unit: Optional[str]) -> str:
    """Normalize a dietary-record unit for portion matching."""
    if unit is None:
        return ""
    unit_text = re.sub(r"\s+", " ", str(unit).strip())
    if not unit_text:
        return ""
    direct = UNIT_SYNONYMS.get(unit_text)
    if direct:
        return direct
    lower = unit_text.lower().replace(".", "")
    lower = re.sub(r"\s+", " ", lower).strip()
    return UNIT_SYNONYMS.get(lower, lower)


def parse_amount_value(amount: Any) -> Optional[float]:
    """
    Parse common handwritten amount strings into a float.

    Supports decimals, simple fractions, and mixed fractions such as "1 1/2".
    Returns None when the amount cannot be converted safely.
    """
    if amount is None:
        return None
    text = str(amount).strip().replace(",", "")
    if not text:
        return None
    text = text.replace("½", " 1/2").replace("¼", " 1/4").replace("¾", " 3/4")
    text = re.sub(r"\s+", " ", text)

    parts = text.split()
    try:
        if len(parts) == 1:
            if "/" in parts[0]:
                numerator, denominator = parts[0].split("/", 1)
                return float(numerator) / float(denominator)
            return float(parts[0])
        if len(parts) == 2 and "/" in parts[1]:
            numerator, denominator = parts[1].split("/", 1)
            return float(parts[0]) + (float(numerator) / float(denominator))
    except (ValueError, ZeroDivisionError):
        return None
    return None


def get_food_portions(conn: sqlite3.Connection, food_code: Optional[str]) -> List[Dict[str, Any]]:
    """Return FNDDS portions and gram weights available for one food_code."""
    if not food_code:
        return []
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT
            fw.Portion_code,
            fpd.Portion_description,
            fw.Portion_weight
        FROM foodweights fw
        LEFT JOIN foodportiondesc fpd
            ON CAST(fw.Portion_code AS TEXT) = CAST(fpd.Portion_code AS TEXT)
        WHERE CAST(fw.Food_code AS TEXT) = ?
        ORDER BY fw.Seq_num
        """,
        (str(food_code),),
    )
    portions = []
    for portion_code, portion_description, portion_weight in cursor.fetchall():
        portions.append({
            "portion_code": str(portion_code) if portion_code is not None else None,
            "portion_description": portion_description or "",
            "portion_weight": float(portion_weight) if portion_weight is not None else None,
        })
    return portions


def extract_portion_unit_quantities(portion_description: str) -> List[Tuple[str, float]]:
    """
    Extract measurable unit quantities from an FNDDS portion description.

    Examples:
    - "1 tablespoon" -> [("tablespoon", 1)]
    - "1 espresso cup (2 fl oz)" -> [("cup", 1), ("fl oz", 2)]
    """
    if not portion_description:
        return []
    desc = portion_description.lower().replace(".", "")
    desc = desc.replace("½", " 1/2").replace("¼", " 1/4").replace("¾", " 3/4")
    quantities: List[Tuple[str, float]] = []
    for canonical_unit, pattern in PORTION_UNIT_PATTERNS:
        for match in re.finditer(pattern, desc):
            prefix = desc[:match.start()].strip()
            number_match = re.search(r"(\d+(?:\.\d+)?(?:\s+\d+/\d+)?|\d+/\d+)\s*$", prefix)
            quantity = parse_amount_value(number_match.group(1)) if number_match else None
            if quantity is not None and quantity > 0:
                quantities.append((canonical_unit, quantity))
    return quantities


def score_portion_for_unit(portion: Dict[str, Any], unit: str) -> float:
    """Score how well a portion description matches a record unit."""
    normalized_unit = normalize_unit(unit)
    if not normalized_unit:
        return 0.0
    desc = (portion.get("portion_description") or "").lower()
    score = 0.0
    if normalized_unit in MASS_UNIT_TO_GRAMS and "quantity not specified" in desc:
        score += 50.0
    if any(unit_name == normalized_unit for unit_name, _ in extract_portion_unit_quantities(desc)):
        score += 100.0
    if normalized_unit and normalized_unit in desc:
        score += 20.0
    if normalized_unit not in MASS_UNIT_TO_GRAMS and "quantity not specified" in desc:
        score -= 10.0
    if portion.get("portion_weight") in (None, 0):
        score -= 5.0
    return score


def select_portion_with_gpt(
    food: Dict[str, Any],
    selected_match: Optional[Dict[str, Any]],
    portions: List[Dict[str, Any]],
    client: Optional["OpenAI"] = None,
) -> Optional[str]:
    """Ask GPT to choose the best FNDDS portion_code for the record amount/unit."""
    if not portions or not OPENAI_AVAILABLE or client is None:
        return None

    portions_text = "\n".join(
        f"- portion_code: {p['portion_code']} | {p['portion_description']} | portion_weight_g: {p.get('portion_weight')}"
        for p in portions
    )
    prompt = f"""You are mapping a dietary-record amount/unit to an FNDDS portion for a selected food.

Selected FNDDS food:
food_code: {selected_match.get('food_code') if selected_match else ''}
description: {selected_match.get('main_food_description') if selected_match else ''}

Dietary record item:
food_description: {food.get('food_description', '')}
amount: {food.get('amount', '')}
original_unit: {food.get('unit', '')}
normalized_unit: {food.get('normalized_unit', food.get('unit', ''))}

Available FNDDS portions for this selected food:
{portions_text}

Choose the ONE portion_code whose portion description best corresponds to the dietary-record unit/measure. Prefer an exact unit match such as tablespoon to tablespoon, fl oz to fl oz, cup to cup, slice to slice. If there is no exact unit match, choose the closest usable household portion for this food. Return ONLY the portion_code."""

    try:
        response = client.chat.completions.create(
            model="gpt-5.2",
            messages=[
                {"role": "system", "content": "You select the single best FNDDS portion_code for amount/unit conversion. Reply with only the portion_code."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_completion_tokens=50,
        )
        content = (response.choices[0].message.content or "").strip()
        code = content.split()[0] if content else ""
        valid_codes = {str(p["portion_code"]) for p in portions}
        return code if code in valid_codes else None
    except Exception as e:
        print(f"  ⚠️  GPT portion selection error: {e}")
        return None


def select_portion_for_food(
    food: Dict[str, Any],
    selected_match: Optional[Dict[str, Any]],
    portions: List[Dict[str, Any]],
    client: Optional["OpenAI"] = None,
) -> Optional[Dict[str, Any]]:
    """Select one FNDDS portion for the selected food using GPT, with heuristic fallback."""
    if not portions:
        return None

    selected_code = select_portion_with_gpt(food, selected_match, portions, client)
    if selected_code:
        return next((p for p in portions if str(p.get("portion_code")) == selected_code), None)

    unit_candidates = [
        food.get("normalized_unit", ""),
        food.get("unit", ""),
    ]
    best_portion = None
    best_score = float("-inf")
    for portion in portions:
        score = max(score_portion_for_unit(portion, unit) for unit in unit_candidates)
        if score > best_score:
            best_score = score
            best_portion = portion
    return best_portion or portions[0]


def calculate_grams(
    amount: Any,
    unit: Optional[str],
    portion: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Convert original amount/unit to grams using selected FNDDS portion information."""
    amount_value = parse_amount_value(amount)
    normalized_unit = normalize_unit(unit)
    result = {
        "amount_numeric": amount_value,
        "grams": None,
        "conversion_method": None,
        "portion_unit_quantity": None,
        "portion_unit": None,
    }
    if amount_value is None:
        result["conversion_method"] = "amount_not_numeric"
        return result

    if not portion or portion.get("portion_weight") in (None, 0):
        if normalized_unit in MASS_UNIT_TO_GRAMS:
            result["grams"] = amount_value * MASS_UNIT_TO_GRAMS[normalized_unit]
            result["conversion_method"] = "direct_mass_unit"
            return result
        result["conversion_method"] = "missing_portion_weight"
        return result

    portion_weight = float(portion["portion_weight"])
    quantities = extract_portion_unit_quantities(portion.get("portion_description", ""))
    matching_quantities = [(u, q) for u, q in quantities if u == normalized_unit]
    if matching_quantities:
        portion_unit, portion_quantity = matching_quantities[0]
        result["grams"] = amount_value * portion_weight / portion_quantity
        result["conversion_method"] = "portion_unit_ratio"
        result["portion_unit_quantity"] = portion_quantity
        result["portion_unit"] = portion_unit
        return result

    equivalent_quantities = [
        (u, q)
        for u, q in quantities
        if normalized_unit in VOLUME_UNIT_IN_TABLESPOONS and u in VOLUME_UNIT_IN_TABLESPOONS
    ]
    if equivalent_quantities:
        portion_unit, portion_quantity = equivalent_quantities[0]
        record_tablespoons = amount_value * VOLUME_UNIT_IN_TABLESPOONS[normalized_unit]
        portion_tablespoons = portion_quantity * VOLUME_UNIT_IN_TABLESPOONS[portion_unit]
        if portion_tablespoons > 0:
            result["grams"] = record_tablespoons * portion_weight / portion_tablespoons
            result["conversion_method"] = "equivalent_volume_unit_ratio"
            result["portion_unit_quantity"] = portion_quantity
            result["portion_unit"] = portion_unit
            return result

    if normalized_unit in MASS_UNIT_TO_GRAMS:
        result["grams"] = amount_value * MASS_UNIT_TO_GRAMS[normalized_unit]
        result["conversion_method"] = "direct_mass_unit"
        return result

    result["grams"] = amount_value * portion_weight
    result["conversion_method"] = "amount_times_selected_portion_weight"
    return result


def build_food_match_data(
    food: Dict[str, Any],
    matches: List[Dict[str, Any]],
    gpt_selected_food_code: Optional[str],
) -> Dict[str, Any]:
    """Build the food-code match payload before portion mapping is added."""
    if "normalized_description" in food:
        normalized_desc = normalize_text(food["normalized_description"])
    else:
        normalized_desc = normalize_text(food["food_description"])

    match_data = {
        "food_description": food["food_description"],  # raw text
        "normalized_description": normalized_desc,
        "amount": food["amount"],
        "unit": food["unit"],
        "normalized_unit": food.get("normalized_unit", food.get("unit")),
        "matches": matches,
        "gpt_selected_food_code": gpt_selected_food_code,
    }

    # Include GPT normalization fields if available
    if "core_food" in food:
        match_data["core_food"] = food["core_food"]
    if "modifiers" in food:
        match_data["modifiers"] = food["modifiers"]
    if "process_method" in food:
        match_data["process_method"] = food["process_method"]

    return match_data


def add_portion_mapping_to_food_match(
    food_match: Dict[str, Any],
    normalized_food: Optional[Dict[str, Any]],
    conn: sqlite3.Connection,
    client: Optional["OpenAI"] = None,
) -> Dict[str, Any]:
    """
    Add selected FNDDS portion fields and grams to one food match object.

    This is deliberately separate from food-code matching so future flags can rerun
    portion mapping without recomputing embeddings or GPT food-code choices.
    """
    normalized_food = normalized_food or {}
    food_match["normalized_unit"] = normalized_food.get(
        "normalized_unit",
        food_match.get("normalized_unit", food_match.get("unit")),
    )

    selected_code = food_match.get("gpt_selected_food_code")
    selected_match = next(
        (m for m in food_match.get("matches", []) if m.get("food_code") == selected_code),
        None,
    )
    food_for_portion = {
        "food_description": food_match.get("food_description"),
        "amount": food_match.get("amount"),
        "unit": food_match.get("unit"),
        "normalized_unit": food_match.get("normalized_unit"),
    }

    portions = get_food_portions(conn, selected_code)
    selected_portion = select_portion_for_food(
        food=food_for_portion,
        selected_match=selected_match,
        portions=portions,
        client=client,
    )
    gram_conversion = calculate_grams(
        amount=food_match.get("amount"),
        unit=food_match.get("normalized_unit") or food_match.get("unit"),
        portion=selected_portion,
    )

    food_match.update({
        "mapped_portion_code": selected_portion.get("portion_code") if selected_portion else None,
        "mapped_portion_description": selected_portion.get("portion_description") if selected_portion else None,
        "mapped_portion_weight": selected_portion.get("portion_weight") if selected_portion else None,
        "grams": gram_conversion["grams"],
        "gram_conversion_method": gram_conversion["conversion_method"],
        "amount_numeric": gram_conversion["amount_numeric"],
        "portion_unit": gram_conversion["portion_unit"],
        "portion_unit_quantity": gram_conversion["portion_unit_quantity"],
    })
    return food_match


def load_normalized_foods_for_match(
    match_data: Dict[str, Any],
    normalized_records_dir: Path,
) -> List[Dict[str, Any]]:
    """Load normalized foods matching a record's match JSON, if available."""
    record_id = match_data.get("record_id")
    if not record_id:
        return []
    record_path = normalized_records_dir / f"{record_id}_record.json"
    if not record_path.exists():
        return []
    try:
        with open(record_path, "r", encoding="utf-8") as f:
            return json.load(f).get("foods", [])
    except (OSError, json.JSONDecodeError):
        return []


def nutrient_key(nutrient: Dict[str, Any]) -> str:
    """Return a stable, readable key for a nutrient entry."""
    return nutrient.get("tagname") or f"nutrient_{nutrient['nutrient_code']}"


def get_food_nutrients(
    conn: sqlite3.Connection,
    food_code: Optional[str],
) -> List[Dict[str, Any]]:
    """Return FNDDS nutrient values per 100 g for one food_code."""
    if not food_code:
        return []
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT
            fn.Nutrient_code,
            nd.Nutrient_description,
            nd.Tagname,
            nd.Unit,
            nd.Decimals,
            fn.Nutrient_value
        FROM fnddsnutval fn
        LEFT JOIN nutdesc nd
            ON CAST(fn.Nutrient_code AS TEXT) = CAST(nd.Nutrient_code AS TEXT)
        WHERE CAST(fn.Food_code AS TEXT) = ?
        ORDER BY CAST(fn.Nutrient_code AS INTEGER)
        """,
        (str(food_code),),
    )
    nutrients = []
    for code, description, tagname, unit, decimals, value_per_100g in cursor.fetchall():
        nutrients.append({
            "nutrient_code": str(code),
            "description": description or "",
            "tagname": tagname or "",
            "unit": unit or "",
            "decimals": int(decimals) if decimals is not None else None,
            "value_per_100g": float(value_per_100g) if value_per_100g is not None else None,
        })
    return nutrients


def calculate_nutrients_for_food(
    food_match: Dict[str, Any],
    nutrients_per_100g: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Calculate nutrient amounts for one matched food using its grams."""
    grams = food_match.get("grams")
    grams_value = float(grams) if isinstance(grams, (int, float)) else None
    calculated = {}
    for nutrient in nutrients_per_100g:
        value_per_100g = nutrient.get("value_per_100g")
        amount = None
        if grams_value is not None and value_per_100g is not None:
            amount = round(grams_value * value_per_100g / 100.0, 6)

        key = nutrient_key(nutrient)
        calculated[key] = {
            "nutrient_code": nutrient["nutrient_code"],
            "description": nutrient["description"],
            "tagname": nutrient["tagname"],
            "unit": nutrient["unit"],
            "decimals": nutrient["decimals"],
            "value_per_100g": value_per_100g,
            "amount": amount,
        }
    return calculated


def add_nutrients_to_daily_totals(
    daily_totals: Dict[str, Dict[str, Any]],
    food_nutrients: Dict[str, Dict[str, Any]],
) -> None:
    """Accumulate one food's nutrients into record-level daily totals."""
    for key, nutrient in food_nutrients.items():
        if key not in daily_totals:
            daily_totals[key] = {
                "nutrient_code": nutrient["nutrient_code"],
                "description": nutrient["description"],
                "tagname": nutrient["tagname"],
                "unit": nutrient["unit"],
                "amount": 0.0,
            }
        if nutrient.get("amount") is not None:
            daily_totals[key]["amount"] = round(
                daily_totals[key]["amount"] + nutrient["amount"],
                6,
            )


def parse_time_to_minutes(time_text: Optional[str]) -> Optional[int]:
    """Parse record time text such as '10:00 AM' into minutes after midnight."""
    if not time_text:
        return None
    match = re.search(r"(\d{1,2})(?::(\d{2}))?\s*(AM|PM)?", str(time_text).strip(), re.IGNORECASE)
    if not match:
        return None
    hour = int(match.group(1))
    minute = int(match.group(2) or 0)
    ampm = (match.group(3) or "").upper()
    if ampm == "AM":
        if hour == 12:
            hour = 0
    elif ampm == "PM":
        if hour != 12:
            hour += 12
    if hour > 23 or minute > 59:
        return None
    return hour * 60 + minute


def classify_meal(time_text: Optional[str]) -> str:
    """
    Classify eating time into a simple three-meal bucket.

    Breakfast: before 11:00
    Lunch: 11:00 through 16:59
    Dinner: 17:00 and later
    """
    minutes = parse_time_to_minutes(time_text)
    if minutes is None:
        return "Unspecified"
    if minutes < 11 * 60:
        return "Breakfast"
    if minutes < 17 * 60:
        return "Lunch"
    return "Dinner"


def empty_meal_totals() -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Create stable meal buckets for report summaries."""
    return {
        "Breakfast": {},
        "Lunch": {},
        "Dinner": {},
        "Unspecified": {},
    }


def create_fndds_food_index(db_path: Path) -> None:
    """
    Step 1: Create fndds_food_index table with normalized food descriptions.
    
    This function:
    - Reads mainfooddesc table from FNDDS database
    - Normalizes food descriptions
    - Creates fndds_food_index table
    - Populates it with normalized data
    
    Args:
        db_path: Path to the FNDDS SQLite database
    """
    print("=" * 80)
    print("Step 1: Creating fndds_food_index table")
    print("=" * 80)
    
    if not db_path.exists():
        raise FileNotFoundError(f"FNDDS database not found: {db_path}")
    
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    try:
        # Create fndds_food_index table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS fndds_food_index (
                food_code TEXT PRIMARY KEY,
                main_food_description TEXT NOT NULL,
                normalized_food_description TEXT NOT NULL,
                source_version TEXT NOT NULL,
                normalize_version TEXT NOT NULL
            )
        """)
        
        # Create index on normalized description for faster search
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_normalized_description 
            ON fndds_food_index(normalized_food_description)
        """)
        
        # Check if table already has data
        cursor.execute("SELECT COUNT(*) FROM fndds_food_index")
        existing_count = cursor.fetchone()[0]
        
        if existing_count > 0:
            print(f"⚠️  Table already contains {existing_count} entries. Clearing...")
            cursor.execute("DELETE FROM fndds_food_index")
        
        # Read all food descriptions from mainfooddesc
        cursor.execute("""
            SELECT Food_code, Main_food_description 
            FROM mainfooddesc
        """)
        
        rows = cursor.fetchall()
        print(f"Found {len(rows)} food descriptions to normalize")
        
        # Normalize and insert
        entries = []
        for food_code, main_desc in rows:
            normalized = normalize_text(main_desc)
            entries.append((
                str(food_code),
                main_desc,
                normalized,
                SOURCE_VERSION,
                NORMALIZE_VERSION
            ))
        
        # Bulk insert into fndds_food_index
        cursor.executemany("""
            INSERT INTO fndds_food_index 
            (food_code, main_food_description, normalized_food_description, 
             source_version, normalize_version)
            VALUES (?, ?, ?, ?, ?)
        """, entries)
        
        conn.commit()
        print(f"✓ Created fndds_food_index with {len(entries)} entries")
        
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()


def build_fndds_embeddings(db_path: Path, embeddings_path: Path, model_name: str = EMBEDDING_MODEL_NAME) -> Path:
    """
    Precompute embeddings for FNDDS normalized food descriptions and save as numpy file.
    
    Uses sentence-transformers/all-MiniLM-L6-v2. The output .npz contains:
    - 'food_codes': shape (N,) array of str, FNDDS food_code
    - 'embeddings': shape (N, 384) float32, one vector per row
    
    Enables fast lookup by food_code during rerank.
    
    Args:
        db_path: Path to FNDDS SQLite database (must have fndds_food_index).
        embeddings_path: Path to save .npz file (e.g. db/fndds/fndds_embeddings.npz).
        model_name: SentenceTransformer model name.
        
    Returns:
        Path to the written embeddings file.
    """
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        raise RuntimeError(
            "sentence_transformers is required for embedding precompute. "
            "Install with: pip install sentence-transformers"
        )
    print("=" * 80)
    print("Step 2: Precomputing FNDDS embeddings (sentence-transformers)")
    print("=" * 80)
    if not db_path.exists():
        raise FileNotFoundError(f"FNDDS database not found: {db_path}")
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute(
        "SELECT food_code, normalized_food_description FROM fndds_food_index ORDER BY food_code"
    )
    rows = cursor.fetchall()
    conn.close()
    food_codes = np.array([str(r[0]) for r in rows], dtype=object)
    descriptions = [r[1] for r in rows]
    print(f"Encoding {len(descriptions)} normalized FNDDS descriptions with {model_name}...")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(descriptions, show_progress_bar=True, convert_to_numpy=True)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    embeddings_path = Path(embeddings_path)
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(embeddings_path, food_codes=food_codes, embeddings=embeddings)
    print(f"✓ Saved embeddings to {embeddings_path}")
    return embeddings_path


def load_fndds_embeddings(embeddings_path: Path) -> Dict[str, np.ndarray]:
    """
    Load precomputed FNDDS embeddings from .npz into a dict food_code -> embedding.
    
    Args:
        embeddings_path: Path to .npz from build_fndds_embeddings.
        
    Returns:
        Dict mapping food_code (str) to embedding vector (np.ndarray, shape (384,)).
    """
    data = np.load(embeddings_path, allow_pickle=True)
    food_codes = data["food_codes"]
    embeddings = data["embeddings"]
    return {str(c): embeddings[i] for i, c in enumerate(food_codes)}


def rank_all_by_embeddings(
    query_text: str,
    all_food_codes: List[str],
    all_food_descriptions: Dict[str, str],
    model: "SentenceTransformer",
    embeddings_by_code: Dict[str, np.ndarray],
    top_k: int = EMBEDDING_RERANK_TOP_K,
) -> List[Dict[str, Any]]:
    """
    Rank ALL FNDDS candidates by embedding similarity.
    
    Uses normalized food description from record as query and precomputed FNDDS embeddings.
    Computes cosine similarity for all candidates and returns top_k.
    
    Args:
        query_text: GPT-normalized food description from record (used as query).
        all_food_codes: List of all FNDDS food codes to rank.
        all_food_descriptions: Dict food_code -> main_food_description.
        model: Loaded SentenceTransformer model.
        embeddings_by_code: Dict food_code -> embedding from load_fndds_embeddings.
        top_k: Number of top results to return (default 10).
        
    Returns:
        List of dicts with keys: food_code, main_food_description, similarity_score.
    """
    if not all_food_codes:
        return []
    
    # Encode query
    query_emb = model.encode([query_text.strip() or " "], convert_to_numpy=True)
    query_emb = np.asarray(query_emb, dtype=np.float32)
    
    # Get embeddings for all candidates
    cand_embs = []
    valid_codes = []
    for code in all_food_codes:
        if code in embeddings_by_code:
            cand_embs.append(embeddings_by_code[code])
            valid_codes.append(code)
        # Skip codes without embeddings (shouldn't happen if embeddings are complete)
    
    if not cand_embs:
        return []
    
    cand_embs = np.stack(cand_embs)
    
    # Cosine similarity: (query @ cand.T) / (||query|| * ||cand||)
    qn = np.linalg.norm(query_emb, axis=1, keepdims=True)
    cn = np.linalg.norm(cand_embs, axis=1, keepdims=True)
    sims = (query_emb @ cand_embs.T).ravel() / (qn.ravel() * cn.ravel() + 1e-9)
    
    # Sort by similarity (descending) and return top_k
    order = np.argsort(-sims)
    out = []
    for idx in order[:top_k]:
        code = valid_codes[idx]
        out.append({
            "food_code": code,
            "main_food_description": all_food_descriptions.get(code, "Unknown"),
            "similarity_score": float(sims[idx]),
        })
    return out


def select_final_candidate_with_gpt(
    original_text: str,
    normalized_description: str,
    top10_matches: List[Dict[str, Any]],
    client: Optional["OpenAI"] = None,
) -> Optional[str]:
    """
    Ask GPT to select one best-matching FNDDS candidate from the top 10 by meaning.
    
    Args:
        original_text: Original food description from the record.
        normalized_description: Normalized form used for embedding.
        top10_matches: List of dicts with food_code, main_food_description, similarity_score.
        client: OpenAI client (created if None and API available).
        
    Returns:
        food_code of the selected candidate, or None if unavailable/failed.
    """
    if not top10_matches or not OPENAI_AVAILABLE:
        return None
    try:
        _client = client or OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    except Exception:
        return None
    if _client is None:
        return None

    candidates_text = "\n".join(
        f"- food_code: {m['food_code']} | {m['main_food_description']} (similarity: {m.get('similarity_score', 0):.4f})"
        for m in top10_matches
    )
    prompt = f"""You are matching a dietary record food to the FNDDS database.

Original food description (from record): "{original_text}"
Normalized form: "{normalized_description}"

Top 10 FNDDS candidates (by embedding similarity):

{candidates_text}

Choose the ONE candidate that best matches the MEANING of the original food description. Consider the main food, modifiers, and preparation context. Return ONLY the food_code of your choice, nothing else (e.g. 16756000)."""

    try:
        response = _client.chat.completions.create(
            model="gpt-5.2",
            messages=[
                {"role": "system", "content": "You select the single best FNDDS match by meaning. Reply with only the food_code."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_completion_tokens=50,
        )
        content = (response.choices[0].message.content or "").strip()
        # Extract food_code: allow digits only or trailing/leading whitespace
        code = content.split()[0] if content else ""
        if not code.isdigit():
            return None
        valid_codes = {m["food_code"] for m in top10_matches}
        return code if code in valid_codes else None
    except Exception as e:
        print(f"  ⚠️  GPT selection error: {e}")
        return None


def parse_reviewed_html(html_path: Path) -> Optional[RecordData]:
    """
    Step 3 (parse): Parse reviewed HTML file and extract record data.
    
    Args:
        html_path: Path to reviewed HTML file
        
    Returns:
        RecordData object with metadata and food items, or None if parsing fails
    """
    if not html_path.exists():
        return None
    
    with open(html_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Extract metadata
    metadata = {
        "record_id": "",
        "subject_id": "",
        "day_of_week": "",
        "diet_type": "",
        "date_of_record": "",
        "reviewer_id": "",
    }
    
    for p in soup.find_all("p"):
        strong = p.find("strong")
        if not strong:
            continue
        label = strong.get_text(strip=True).replace(":", "").lower()
        value = p.get_text(strip=True).replace(strong.get_text(strip=True), "").strip()
        
        if "record id" in label:
            metadata["record_id"] = value
        elif "subject id" in label:
            metadata["subject_id"] = value
        elif "day of week" in label:
            metadata["day_of_week"] = value
        elif "diet" in label:
            metadata["diet_type"] = value
        elif "date of record" in label:
            metadata["date_of_record"] = value
        elif "reviewer id" in label:
            metadata["reviewer_id"] = value
    
    # Extract food items from table
    table = soup.find("table")
    if not table:
        return None
    
    tbody = table.find("tbody") or table
    rows = tbody.find_all("tr")
    
    foods = []
    for row in rows:
        cells = row.find_all("td")
        if len(cells) < 5:
            continue
        
        # Check if first column is Flag column (6 columns) or regular (5 columns)
        has_flag_column = len(cells) >= 6
        data_start = 1 if has_flag_column else 0
        
        # Extract: time, place, amount, measure, description
        if len(cells) < data_start + 5:
            continue
        
        time = cells[data_start].get_text(strip=True)
        place = cells[data_start + 1].get_text(strip=True)
        amount = cells[data_start + 2].get_text(strip=True)
        measure = cells[data_start + 3].get_text(strip=True)
        description = cells[data_start + 4].get_text(strip=True)
        
        # Skip empty rows
        if not description:
            continue
        
        # Parse amount and unit (simple heuristic: split on space)
        amount_parts = amount.split(maxsplit=1)
        food_amount = amount_parts[0] if amount_parts else amount
        food_unit = amount_parts[1] if len(amount_parts) > 1 else measure
        
        foods.append(RecordFoodItem(
            time=time,
            place=place,
            food_description=description,
            amount=food_amount,
            unit=food_unit
        ))
    
    return RecordData(
        record_id=metadata["record_id"],
        subject_id=metadata["subject_id"],
        day_of_week=metadata["day_of_week"],
        diet_type=metadata["diet_type"],
        date_of_record=metadata["date_of_record"],
        reviewer_id=metadata["reviewer_id"],
        foods=foods
    )


def process_records_to_json(review_dir: Path, output_dir: Path) -> List[Path]:
    """
    Step 3: Process all reviewed HTML files and generate JSON objects.
    
    Args:
        review_dir: Directory containing reviewed HTML files
        output_dir: Directory to save JSON output files
        
    Returns:
        List of paths to generated JSON files
    """
    print("\n" + "=" * 80)
    print("Step 2: Processing records to JSON")
    print("=" * 80)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    html_files = sorted(review_dir.glob("*_Reviewed.html"))
    print(f"Found {len(html_files)} reviewed HTML files")
    
    json_files = []
    for html_file in html_files:
        print(f"Processing {html_file.name}...", end=" ")
        record_data = parse_reviewed_html(html_file)
        
        if not record_data:
            print("⚠️  Failed to parse")
            continue
        
        # Convert to JSON-serializable format
        json_data = {
            "record_id": record_data.record_id,
            "metadata": {
                "subject_id": record_data.subject_id,
                "day_of_week": record_data.day_of_week,
                "diet_type": record_data.diet_type,
                "date_of_record": record_data.date_of_record,
                "reviewer_id": record_data.reviewer_id,
            },
            "foods": [
                {
                    "time": food.time,
                    "place": food.place,
                    "food_description": food.food_description,
                    "amount": food.amount,
                    "unit": food.unit,
                }
                for food in record_data.foods
            ]
        }
        
        # Save JSON file
        json_path = output_dir / f"{record_data.record_id}_record.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        json_files.append(json_path)
        print(f"✓ {len(record_data.foods)} food items")
    
    print(f"\n✓ Generated {len(json_files)} JSON files")
    return json_files


def normalize_food_item_with_gpt(
    food_description: str,
    amount: str,
    unit: str,
    client: Optional[OpenAI] = None
) -> Dict[str, Any]:
    """
    Normalize a food item using GPT to extract structured information.
    
    Main aim: Remove ambiguity so the description matches FNDDS well. When one food
    is used as a modifier for another, the main food stays as the semantic head and
    the modifier food is expressed as a modifier-sound term. normalized_description
    is used for embedding similarity and ranking.
    
    Extracts:
    - core_food: The main food item in singular form (the primary food, not the modifier)
    - modifiers: Modifier terms in singular form; when a food modifies another,
      express it as a modifier-sound word (e.g., "chocolate" for chocolate cake)
    - process_method: Cooking/preparation method if mentioned
    - normalized_description: Unambiguous phrase with main food as semantic head,
      quantity words removed, singular form; used for embedding similarity
    
    All output fields use singular forms.

    Args:
        food_description: Original food description
        amount: Amount value
        unit: Unit of measurement
        client: OpenAI client instance (created if None)

    Returns:
        Dictionary with normalized fields (normalized_description is used for embedding similarity).
    """
    if not OPENAI_AVAILABLE:
        # Fallback: return basic normalization without GPT
        return {
            "normalized_description": food_description,
            "core_food": food_description,
            "modifiers": [],
            "process_method": None,
            "normalized_unit": unit
        }
    
    if client is None:
        client = create_openai_client()
    
    # Create prompt for GPT normalization
    prompt = f"""Analyze the following food description and extract structured information.

Food Description: "{food_description}"
Amount: {amount}
Unit: {unit}

MAIN AIM: Remove ambiguity for matching to a food database. When one food is used as a modifier for another (e.g., "chocolate cake", "peach yogurt", "raisin bread"), identify the MAIN food (cake, yogurt, bread) and the MODIFIER food (chocolate, peach, raisin). The main food must stay as the semantic head; the modifier food must be expressed as a modifier-sound term so it is clear what is the primary food.

Extract the following:
1. core_food: The MAIN food item in SINGULAR form — the primary food that is being described (e.g., "cake", "yogurt", "bread", "milk", "coffee"). When a food modifies another, the head food is core_food (e.g., for "chocolate cake" use core_food "cake", not "chocolate").
2. modifiers: A list of modifier terms in SINGULAR form. When a food word modifies the main food, include it as a modifier-sound term (e.g., ["chocolate"], ["peach"], ["raisin"]). Also include non-food modifiers like ["deli", "thin slice", "sparkling"]. Return as a JSON array.
3. process_method: Cooking or preparation method if mentioned (e.g., "baked", "grilled", "raw"). Return null if not mentioned.
4. normalized_description: A single UNAMBIGUOUS phrase used for embedding similarity and ranking. Requirements:
   - Remove all quantity words and numbers (amount/unit are separate).
   - Use SINGULAR forms throughout.
   - The MAIN food must be the semantic head; modifier foods must read as modifiers (e.g., "chocolate cake", "peach yogurt", "raisin bread", "thin slice turkey"). This string will be used for embedding similarity — keep it clear and UNAMBIGUOUS so the main food is identifiable.
   - PUNCTUATION: Keep percent (%), decimals (e.g. 3.25), and fractions (e.g. 1/2). Replace "+" with "and". Remove all other punctuation (commas, hyphens, apostrophes, etc.).
5. normalized_unit: Normalize the unit:
   - "T", "tbsp" -> "tablespoon", "t", "tsp" -> "teaspoon"
   - For fluids (coffee, water, juice, milk, etc.) and unit "oz" -> "fl oz"
   - Otherwise keep as-is.
6. Keep all words in lowercase.

Return ONLY a valid JSON object with these exact keys:
{{
    "core_food": "...",
    "modifiers": [...],
    "process_method": "..." or null,
    "normalized_description": "...",
    "normalized_unit": "..."
}}

No explanations, only the JSON object."""
    
    try:
        response = client.chat.completions.create(
            model="gpt-5.2",
            messages=[
                {"role": "system", "content": "You are a food data normalization expert. Extract structured information from food descriptions and return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_completion_tokens=500
        )
        
        content = response.choices[0].message.content.strip()
        
        # Extract JSON from response (handle cases where GPT adds markdown formatting)
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        
        # Parse JSON response
        result = json.loads(content)
        
        # Ensure all required fields exist
        return {
            "normalized_description": result.get("normalized_description", food_description),
            "core_food": result.get("core_food", food_description),
            "modifiers": result.get("modifiers", []),
            "process_method": result.get("process_method"),
            "normalized_unit": result.get("normalized_unit", unit)
        }
        
    except json.JSONDecodeError as e:
        content_preview = content[:200] if 'content' in locals() else "N/A"
        print(f"  ⚠️  Failed to parse GPT response as JSON: {e}")
        print(f"  Response was: {content_preview}")
        # Fallback to basic normalization
        return {
            "normalized_description": food_description,
            "core_food": food_description,
            "modifiers": [],
            "process_method": None,
            "normalized_unit": unit
        }
    except Exception as e:
        print(f"  ⚠️  Error calling GPT-5.2: {e}")
        if is_incorrect_hostname_error(e):
            print("  Hint: add OPENAI_BASE_URL=https://us.api.openai.com/v1 to your .env and restart the terminal.")
        # Fallback to basic normalization
        return {
            "normalized_description": food_description,
            "core_food": food_description,
            "modifiers": [],
            "process_method": None,
            "normalized_unit": unit
        }


def normalize_records_with_gpt(
    json_files: List[Path],
    output_dir: Path
) -> List[Path]:
    """
    Step 3: Normalize food descriptions in JSON files using GPT-5.2.
    
    For each food item, extracts:
    - core_food: Main food item
    - modifiers: Descriptive modifiers
    - process_method: Cooking/preparation method
    - normalized_description: Description with quantity words removed
    
    Args:
        json_files: List of JSON file paths from step 2
        output_dir: Directory to save normalized JSON files
        
    Returns:
        List of paths to generated normalized JSON files
    """
    print("\n" + "=" * 80)
    print("Step 4: Normalizing food descriptions with GPT-5.2")
    print("=" * 80)
    
    if not OPENAI_AVAILABLE:
        print("⚠️  OpenAI not available. Skipping normalization step.")
        return json_files
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  OPENAI_API_KEY not found. Skipping normalization step.")
        return json_files
    
    # Detect records that already have normalized data (normalized_description present)
    already_normalized_ids = []
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                record_data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        record_id = record_data.get("record_id")
        if not record_id:
            continue
        normalized_path = output_dir / f"{record_id}_record.json"
        if not normalized_path.exists():
            continue
        try:
            with open(normalized_path, 'r', encoding='utf-8') as f:
                existing = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if any(f.get("normalized_description") for f in existing.get("foods", [])):
            already_normalized_ids.append(record_id)
    
    renormalize_existing = True
    if already_normalized_ids:
        prompt = (
            f"{len(already_normalized_ids)} record(s) already have normalized data: "
            f"{', '.join(already_normalized_ids)}. Re-normalize them? [y/N]: "
        )
        reply = input(prompt).strip().lower()
        if reply not in ("y", "yes"):
            renormalize_existing = False
            print("Skipping re-normalization for those records (using existing files).")
    
    client = create_openai_client(api_key)
    normalized_files = []
    
    for json_file in json_files:
        print(f"Processing {json_file.name}...", end=" ")
        
        # Load record JSON
        with open(json_file, 'r', encoding='utf-8') as f:
            record_data = json.load(f)
        
        record_id = record_data["record_id"]
        normalized_path = output_dir / f"{record_id}_record.json"
        if (
            record_id in already_normalized_ids
            and not renormalize_existing
        ):
            normalized_files.append(normalized_path)
            print("skipped (using existing normalized data)")
            continue
        
        # Normalize each food item
        normalized_foods = []
        for food in record_data["foods"]:
            # Normalize using GPT-5.2
            normalized = normalize_food_item_with_gpt(
                food["food_description"],
                food["amount"],
                food["unit"],
                client
            )
            
            # Create normalized food item
            normalized_food = {
                "time": food["time"],
                "place": food["place"],
                "food_description": food["food_description"],
                "normalized_description": normalized["normalized_description"],
                "core_food": normalized["core_food"],
                "modifiers": normalized["modifiers"],
                "process_method": normalized["process_method"],
                "amount": food["amount"],
                "unit": food["unit"],  # original unit
                "normalized_unit": normalized.get("normalized_unit", food["unit"])  # normalized unit
            }
            normalized_foods.append(normalized_food)
        
        # Create normalized record
        normalized_record = {
            "record_id": record_data["record_id"],
            "metadata": record_data["metadata"],
            "foods": normalized_foods
        }
        
        # Save normalized JSON file (overwrite original or create new)
        with open(normalized_path, 'w', encoding='utf-8') as f:
            json.dump(normalized_record, f, indent=2, ensure_ascii=False)
        
        normalized_files.append(normalized_path)
        print(f"✓ {len(normalized_foods)} foods normalized")
    
    print(f"\n✓ Generated {len(normalized_files)} normalized JSON files")
    return normalized_files



def match_records_to_fndds(
    json_files: List[Path],
    db_path: Path,
    output_dir: Path,
    embeddings_path: Optional[Path] = None,
    embedding_model_name: str = EMBEDDING_MODEL_NAME,
) -> List[Path]:
    """
    Step 5: Match record food descriptions to FNDDS (embedding similarity + GPT final selection).
    
    Strategy:
    1. Load FNDDS food codes and descriptions; query = normalized_description.
    2. Encode with sentence-transformers; rank all by cosine similarity, return top 10.
    3. GPT selects one final candidate from the top 10 by meaning match to original text.
    
    Requires precomputed FNDDS embeddings at embeddings_path (from build_fndds_embeddings).
    
    Args:
        json_files: List of normalized JSON file paths from step 4
        db_path: Path to FNDDS database
        output_dir: Directory to save match results
        embeddings_path: Path to fndds_embeddings.npz (default FNDDS_EMBEDDINGS_PATH)
        embedding_model_name: SentenceTransformer model for encoding query and similarity
        
    Returns:
        List of paths to generated match JSON files
    """
    emb_path = Path(embeddings_path or FNDDS_EMBEDDINGS_PATH)
    if not emb_path.exists():
        raise FileNotFoundError(
            f"FNDDS embeddings not found at {emb_path}. "
            "Run build_fndds_embeddings() first (e.g. from main)."
        )
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        raise RuntimeError(
            "sentence_transformers is required for matching. pip install sentence-transformers"
        )
    
    print("\n" + "=" * 80)
    print("Step 5: Matching records to FNDDS (embedding similarity only)")
    print("=" * 80)
    
    # Load all FNDDS food codes and descriptions
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute("SELECT food_code, main_food_description FROM fndds_food_index")
    rows = cursor.fetchall()
    conn.close()
    
    all_food_codes = [str(row[0]) for row in rows]
    all_food_descriptions = {str(row[0]): row[1] for row in rows}
    print(f"Loaded {len(all_food_codes)} FNDDS food items")
    
    print("Loading FNDDS embeddings and embedding model...")
    embeddings_by_code = load_fndds_embeddings(emb_path)
    model = SentenceTransformer(embedding_model_name)
    print(f"Loaded {len(embeddings_by_code)} embeddings, model {embedding_model_name}")

    # OpenAI client for GPT final-candidate selection (optional)
    gpt_client = None
    if OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
        try:
            gpt_client = create_openai_client()
        except Exception:
            pass
    
    match_files = []
    
    for json_file in json_files:
        print(f"Processing {json_file.name}...", end=" ")
        
        # Load record JSON
        with open(json_file, 'r', encoding='utf-8') as f:
            record_data = json.load(f)
        
        # Process each food item
        food_matches = []
        for food in record_data["foods"]:
            # Query = normalized_food_description (normalized_description) for embedding similarity and ranking
            query_text = (food.get("normalized_description") or food.get("food_description") or food.get("core_food") or "").strip()
            
            # Rank all FNDDS candidates by embedding similarity, return top 10
            matches = rank_all_by_embeddings(
                query_text=query_text,
                all_food_codes=all_food_codes,
                all_food_descriptions=all_food_descriptions,
                model=model,
                embeddings_by_code=embeddings_by_code,
                top_k=EMBEDDING_RERANK_TOP_K,
            )
            
            # GPT selects one final candidate from top 10 by meaning
            gpt_selected_food_code = select_final_candidate_with_gpt(
                original_text=food.get("food_description", ""),
                normalized_description=food.get("normalized_description", ""),
                top10_matches=matches,
                client=gpt_client,
            )
            food_matches.append(build_food_match_data(food, matches, gpt_selected_food_code))
        
        # Create match result
        match_result = {
            "record_id": record_data["record_id"],
            "metadata": record_data["metadata"],
            "food_matches": food_matches
        }
        
        # Save match JSON
        match_path = output_dir / f"{record_data['record_id']}_matches.json"
        with open(match_path, 'w', encoding='utf-8') as f:
            json.dump(match_result, f, indent=2, ensure_ascii=False)
        
        match_files.append(match_path)
        print(f"✓ {len(food_matches)} foods matched")
    
    print(f"\n✓ Generated {len(match_files)} match JSON files")
    return match_files


def add_portion_mapping_to_matches(
    match_files: List[Path],
    db_path: Path,
    normalized_records_dir: Path,
) -> List[Path]:
    """
    Step 6: Add FNDDS portion-code mapping and gram conversion to match JSON files.

    This can be rerun independently after food-code matching, which keeps the script
    ready for future flags such as --portion-only or --skip-food-match.
    """
    print("\n" + "=" * 80)
    print("Step 6: Mapping selected FNDDS foods to portions and grams")
    print("=" * 80)

    gpt_client = None
    if OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
        try:
            gpt_client = create_openai_client()
        except Exception:
            pass

    conn = sqlite3.connect(str(db_path))
    try:
        for match_file in match_files:
            print(f"Processing portions for {match_file.name}...", end=" ")
            with open(match_file, "r", encoding="utf-8") as f:
                match_data = json.load(f)

            normalized_foods = load_normalized_foods_for_match(match_data, normalized_records_dir)
            for idx, food_match in enumerate(match_data.get("food_matches", [])):
                normalized_food = normalized_foods[idx] if idx < len(normalized_foods) else None
                add_portion_mapping_to_food_match(
                    food_match=food_match,
                    normalized_food=normalized_food,
                    conn=conn,
                    client=gpt_client,
                )

            with open(match_file, "w", encoding="utf-8") as f:
                json.dump(match_data, f, indent=2, ensure_ascii=False)
            print(f"✓ {len(match_data.get('food_matches', []))} foods enriched")
    finally:
        conn.close()

    print(f"\n✓ Added portion mapping to {len(match_files)} match files")
    return match_files


def generate_visualization_html(
    match_files: List[Path],
    output_dir: Path,
    top_k: int = EMBEDDING_RERANK_TOP_K,
) -> List[Path]:
    """
    Step 7: Generate visualization HTML for each record's matches.
    
    GPT selection is shown first (with badge and similarity); then top 10 matches with similarity_score.
    
    Args:
        match_files: List of match JSON file paths
        output_dir: Directory to save HTML files
        top_k: Number of matches shown per food (default 10)
        
    Returns:
        List of paths to generated HTML files
    """
    print("\n" + "=" * 80)
    print("Step 7: Generating visualization HTML (top 10 per food)")
    print("=" * 80)
    
    html_files = []
    
    for match_file in match_files:
        print(f"Generating HTML for {match_file.name}...", end=" ")
        
        # Load match data
        with open(match_file, 'r', encoding='utf-8') as f:
            match_data = json.load(f)
        
        # Generate HTML
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8"/>
    <title>FNDDS Matches - {match_data['record_id']}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }}
        .metadata {{
            background-color: #f9f9f9;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .metadata p {{
            margin: 5px 0;
        }}
        .food-item {{
            margin-bottom: 30px;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 15px;
            background-color: #fafafa;
        }}
        .food-header {{
            font-size: 1.2em;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 10px;
        }}
        .food-details {{
            color: #666;
            font-size: 0.9em;
            margin-bottom: 15px;
        }}
        .matches {{
            margin-top: 15px;
        }}
        .match-item {{
            background-color: white;
            border-left: 4px solid #4CAF50;
            padding: 10px;
            margin-bottom: 10px;
            border-radius: 3px;
        }}
        .match-item.top {{
            border-left-color: #2196F3;
            background-color: #e3f2fd;
        }}
        .match-item.gpt-pick {{
            border-left-color: #9C27B0;
            background-color: #f3e5f5;
        }}
        .gpt-badge {{
            background-color: #9C27B0;
            color: white;
            padding: 2px 8px;
            border-radius: 3px;
            font-size: 0.8em;
            margin-right: 8px;
        }}
        .match-score {{
            float: right;
            background-color: #4CAF50;
            color: white;
            padding: 3px 8px;
            border-radius: 3px;
            font-size: 0.85em;
        }}
        .match-item.top .match-score {{
            background-color: #2196F3;
        }}
        .match-code {{
            font-family: monospace;
            color: #666;
            font-size: 0.9em;
        }}
        .no-matches {{
            color: #999;
            font-style: italic;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>FNDDS Food Matches - {match_data['record_id']}</h1>
        
        <div class="metadata">
            <p><strong>Record ID:</strong> {match_data['record_id']}</p>
            <p><strong>Subject ID:</strong> {match_data['metadata']['subject_id']}</p>
            <p><strong>Date:</strong> {match_data['metadata']['date_of_record']}</p>
            <p><strong>Day of Week:</strong> {match_data['metadata']['day_of_week']}</p>
            <p><strong>Diet Type:</strong> {match_data['metadata']['diet_type']}</p>
            <p><strong>Reviewer ID:</strong> {match_data['metadata']['reviewer_id']}</p>
        </div>
"""
        
        # Add food matches
        for idx, food_match in enumerate(match_data['food_matches']):
            html_content += f"""
        <div class="food-item">
            <div class="food-header">Food Item {idx + 1}</div>
            <div class="food-details">
                <strong>Raw Text:</strong> {food_match['food_description']}<br/>
                <strong>Normalized:</strong> <em>{food_match['normalized_description']}</em><br/>
                <strong>Amount:</strong> {food_match['amount']} {food_match['unit']}<br/>
                <strong>Mapped Portion:</strong> {food_match.get('mapped_portion_code') or 'N/A'} - {food_match.get('mapped_portion_description') or 'N/A'} ({food_match.get('mapped_portion_weight') or 'N/A'} g)<br/>
                <strong>Converted Grams:</strong> {f"{food_match.get('grams'):.2f}" if isinstance(food_match.get('grams'), (int, float)) else 'N/A'} g
            </div>
            <div class="matches">
                <strong>Top {top_k} Matches (embedding similarity)</strong>
"""
            gpt_code = food_match.get('gpt_selected_food_code')
            matches_list = food_match.get('matches') or []
            if matches_list:
                # Show GPT selection first if present, then others in order (skip duplicate in list)
                seen = set()
                if gpt_code:
                    gpt_match = next((m for m in matches_list if m.get('food_code') == gpt_code), None)
                    if gpt_match:
                        seen.add(gpt_code)
                        sim = gpt_match.get('similarity_score')
                        score_str = f"Similarity: {sim:.4f}" if sim is not None else "N/A"
                        html_content += f"""
                <div class="match-item gpt-pick">
                    <span class="gpt-badge">GPT selection</span><span class="match-score">{score_str}</span>
                    <div class="match-code">Food Code: {gpt_match['food_code']}</div>
                    <div>{gpt_match['main_food_description']}</div>
                </div>
"""
                first_remaining = not gpt_code
                for match in matches_list:
                    code = match.get('food_code')
                    if code and code in seen:
                        continue
                    seen.add(code)
                    is_first_remaining = first_remaining
                    first_remaining = False
                    match_class = "match-item top" if is_first_remaining else "match-item"
                    similarity = match.get('similarity_score')
                    score_str = f"Similarity: {similarity:.4f}" if similarity is not None else "N/A"
                    html_content += f"""
                <div class="{match_class}">
                    <span class="match-score">{score_str}</span>
                    <div class="match-code">Food Code: {match['food_code']}</div>
                    <div>{match['main_food_description']}</div>
                </div>
"""
            else:
                html_content += """
                <div class="no-matches">No matches found</div>
"""
            
            html_content += """
            </div>
        </div>
"""
        
        html_content += """
    </div>
</body>
</html>
"""
        
        # Save HTML file
        html_path = output_dir / f"{match_data['record_id']}_matches.html"
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        html_files.append(html_path)
        print(f"✓")
    
    print(f"\n✓ Generated {len(html_files)} visualization HTML files")
    return html_files


def generate_fndds_nutrient_results(
    match_files: List[Path],
    db_path: Path,
    results_dir: Path,
) -> List[Path]:
    """
    Step 8: Calculate FNDDS nutrient amounts from mapped foods and grams.

    FNDDS nutrient values are per 100 g of food. For each matched food:
    nutrient_amount = nutrient_value_per_100g * mapped_grams / 100.
    """
    print("\n" + "=" * 80)
    print("Step 8: Calculating FNDDS nutrient results")
    print("=" * 80)

    results_dir.mkdir(parents=True, exist_ok=True)
    result_files = []
    conn = sqlite3.connect(str(db_path))
    try:
        for match_file in match_files:
            print(f"Calculating nutrients for {match_file.name}...", end=" ")
            with open(match_file, "r", encoding="utf-8") as f:
                match_data = json.load(f)

            food_results = []
            daily_totals: Dict[str, Dict[str, Any]] = {}
            meal_totals = empty_meal_totals()
            normalized_foods = load_normalized_foods_for_match(match_data, match_file.parent)
            for idx, food_match in enumerate(match_data.get("food_matches", [])):
                normalized_food = normalized_foods[idx] if idx < len(normalized_foods) else {}
                time_of_eat = normalized_food.get("time")
                place_of_eat = normalized_food.get("place")
                meal = classify_meal(time_of_eat)
                food_code = food_match.get("gpt_selected_food_code")
                selected_match = next(
                    (m for m in food_match.get("matches", []) if m.get("food_code") == food_code),
                    None,
                )
                nutrients_per_100g = get_food_nutrients(conn, food_code)
                food_nutrients = calculate_nutrients_for_food(food_match, nutrients_per_100g)
                add_nutrients_to_daily_totals(daily_totals, food_nutrients)
                add_nutrients_to_daily_totals(meal_totals.setdefault(meal, {}), food_nutrients)

                food_results.append({
                    "time_of_eat": time_of_eat,
                    "place_of_eat": place_of_eat,
                    "meal": meal,
                    "food_code": food_code,
                    "fndds_food_description": (
                        selected_match.get("main_food_description") if selected_match else None
                    ),
                    "food_amount": {
                        "amount": food_match.get("amount"),
                        "unit": food_match.get("unit"),
                        "normalized_unit": food_match.get("normalized_unit"),
                        "grams": food_match.get("grams"),
                        "gram_conversion_method": food_match.get("gram_conversion_method"),
                        "mapped_portion_code": food_match.get("mapped_portion_code"),
                        "mapped_portion_description": food_match.get("mapped_portion_description"),
                        "mapped_portion_weight": food_match.get("mapped_portion_weight"),
                    },
                    "nutrients": food_nutrients,
                })

            result_data = {
                "record_id": match_data.get("record_id"),
                "subject_id": match_data.get("metadata", {}).get("subject_id"),
                "day_of_week": match_data.get("metadata", {}).get("day_of_week"),
                "diet_type": match_data.get("metadata", {}).get("diet_type"),
                "date_of_record": match_data.get("metadata", {}).get("date_of_record"),
                "reviewer_id": match_data.get("metadata", {}).get("reviewer_id"),
                "source_match_file": str(match_file),
                "nutrient_basis": "FNDDS nutrient values per 100 g; amount = value_per_100g * grams / 100",
                "foods": food_results,
                "daily_nutrient_totals": daily_totals,
                "meal_nutrient_totals": meal_totals,
            }

            result_path = results_dir / f"{match_data.get('record_id')}_fndds_nutrients.json"
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(result_data, f, indent=2, ensure_ascii=False)
            result_files.append(result_path)
            print(f"✓ {len(food_results)} foods, {len(daily_totals)} daily nutrient totals")
    finally:
        conn.close()

    print(f"\n✓ Generated {len(result_files)} FNDDS nutrient result files")
    return result_files


def html_escape(value: Any) -> str:
    """Escape a value for HTML output."""
    if value is None:
        return ""
    return html.escape(str(value))


def format_number(value: Any, digits: int = 2) -> str:
    """Format a numeric value for compact HTML display."""
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return ""


def get_nutrient_display_value(
    nutrients: Dict[str, Dict[str, Any]],
    tagname: Optional[str] = None,
    nutrient_code: Optional[str] = None,
    digits: int = 2,
) -> str:
    """Return one nutrient amount for a food or total by tagname or nutrient code."""
    for key, nutrient in nutrients.items():
        if tagname and (key == tagname or nutrient.get("tagname") == tagname):
            return format_number(nutrient.get("amount"), digits)
        if nutrient_code and nutrient.get("nutrient_code") == nutrient_code:
            return format_number(nutrient.get("amount"), digits)
    return ""


def generate_fndds_results_html(
    result_files: List[Path],
    results_dir: Path,
) -> List[Path]:
    """
    Step 9: Generate readable HTML reports for FNDDS nutrient result JSON files.

    Each record gets one HTML report, and results/fndds/index.html links all records.
    """
    print("\n" + "=" * 80)
    print("Step 9: Generating FNDDS nutrient result HTML reports")
    print("=" * 80)

    results_dir.mkdir(parents=True, exist_ok=True)
    html_files = []
    macro_columns = [
        ("Energy", "ENERC_KCAL", None, "kcal", 0),
        ("Protein", "PROCNT", None, "g", 1),
        ("Fat", "FAT", None, "g", 1),
        ("Carb", "CHOCDF", None, "g", 1),
    ]
    nutrient_columns = [
        *macro_columns,
        ("Fiber", "FIBTG", None, "g", 1),
        ("Sugar", "SUGAR", None, "g", 1),
        ("Sodium", None, "307", "mg", 0),
    ]

    for result_file in result_files:
        print(f"Generating HTML for {result_file.name}...", end=" ")
        with open(result_file, "r", encoding="utf-8") as f:
            result_data = json.load(f)

        record_id = result_data.get("record_id") or result_file.stem
        foods = result_data.get("foods", [])
        daily_totals = result_data.get("daily_nutrient_totals", {})
        meal_totals = result_data.get("meal_nutrient_totals", {})

        rows = []
        for idx, food in enumerate(foods, start=1):
            food_amount = food.get("food_amount", {})
            nutrients = food.get("nutrients", {})
            amount_text = (
                f"{html_escape(food_amount.get('amount'))} "
                f"{html_escape(food_amount.get('unit'))}"
            ).strip()
            nutrient_detail_rows = []
            for key, nutrient in sorted(
                nutrients.items(),
                key=lambda item: int(item[1].get("nutrient_code", "999999")) if str(item[1].get("nutrient_code", "")).isdigit() else 999999,
            ):
                nutrient_detail_rows.append(f"""
                                <tr>
                                    <td class="mono">{html_escape(nutrient.get('nutrient_code'))}</td>
                                    <td>{html_escape(nutrient.get('tagname') or key)}</td>
                                    <td>{html_escape(nutrient.get('description'))}</td>
                                    <td>{html_escape(nutrient.get('unit'))}</td>
                                    <td>{format_number(nutrient.get('value_per_100g'), 6)}</td>
                                    <td>{format_number(nutrient.get('amount'), 6)}</td>
                                </tr>
""")
            rows.append(f"""
                <tr>
                    <td>{idx}</td>
                    <td>{html_escape(food.get('time_of_eat'))}</td>
                    <td>{html_escape(food.get('place_of_eat'))}</td>
                    <td>{html_escape(food.get('meal'))}</td>
                    <td>
                        <strong>{html_escape(food.get('fndds_food_description'))}</strong>
                    </td>
                    <td class="mono">{html_escape(food.get('food_code'))}</td>
                    <td>{amount_text}</td>
                    <td>{format_number(food_amount.get('grams'), 2)}</td>
                    <td>
                        <div class="mono">{html_escape(food_amount.get('mapped_portion_code'))}</div>
                        <div class="muted">{html_escape(food_amount.get('mapped_portion_description'))}</div>
                    </td>
                    <td>{html_escape(food_amount.get('gram_conversion_method'))}</td>
                </tr>
                <tr class="nutrient-detail-row">
                    <td colspan="10">
                        <details>
                            <summary>Food nutrient details</summary>
                            <table class="nested-table">
                                <thead>
                                    <tr>
                                        <th>Nutrient Code</th>
                                        <th>Tag</th>
                                        <th>Description</th>
                                        <th>Unit</th>
                                        <th>Per 100 g</th>
                                        <th>This Food</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {''.join(nutrient_detail_rows)}
                                </tbody>
                            </table>
                        </details>
                    </td>
                </tr>
""")

        total_header_cells = "\n".join(
            f"<th>{html_escape(label)}<br><span>{html_escape(unit)}</span></th>"
            for label, _, _, unit, _ in nutrient_columns
        )
        macro_header_cells = "\n".join(
            f"<th>{html_escape(label)}<br><span>{html_escape(unit)}</span></th>"
            for label, _, _, unit, _ in macro_columns
        )
        meal_summary_rows = []
        for meal in ["Breakfast", "Lunch", "Dinner", "Unspecified"]:
            totals = meal_totals.get(meal, {})
            if meal == "Unspecified" and not totals:
                continue
            macro_value_cells = "\n".join(
                f"<td>{get_nutrient_display_value(totals, tag, code, digits)}</td>"
                for _, tag, code, _, digits in macro_columns
            )
            meal_summary_rows.append(f"""
                    <tr>
                        <td><strong>{html_escape(meal)}</strong></td>
                        {macro_value_cells}
                    </tr>
""")
        daily_macro_cells = "\n".join(
            f"<td>{get_nutrient_display_value(daily_totals, tag, code, digits)}</td>"
            for _, tag, code, _, digits in macro_columns
        )
        meal_summary_rows.append(f"""
                    <tr class="daily-total">
                        <td><strong>Daily Total</strong></td>
                        {daily_macro_cells}
                    </tr>
""")

        all_total_rows = []
        for key, nutrient in sorted(
            daily_totals.items(),
            key=lambda item: int(item[1].get("nutrient_code", "999999")) if str(item[1].get("nutrient_code", "")).isdigit() else 999999,
        ):
            all_total_rows.append(f"""
                <tr>
                    <td class="mono">{html_escape(nutrient.get('nutrient_code'))}</td>
                    <td>{html_escape(nutrient.get('tagname') or key)}</td>
                    <td>{html_escape(nutrient.get('description'))}</td>
                    <td>{html_escape(nutrient.get('unit'))}</td>
                    <td>{format_number(nutrient.get('amount'), 6)}</td>
                </tr>
""")

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8"/>
    <title>FNDDS Nutrient Results - {html_escape(record_id)}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background: #f7f7f7;
            color: #222;
        }}
        .container {{
            max-width: 1500px;
            margin: 0 auto;
            background: #fff;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.08);
        }}
        h1 {{
            margin-top: 0;
            border-bottom: 3px solid #2e7d32;
            padding-bottom: 10px;
        }}
        h2 {{
            margin-top: 28px;
        }}
        .metadata, .totals-strip {{
            background: #f1f8f3;
            border: 1px solid #d8eadc;
            border-radius: 6px;
            padding: 12px;
            margin-bottom: 18px;
        }}
        .metadata span {{
            display: inline-block;
            margin-right: 22px;
            margin-bottom: 6px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 12px;
            background: white;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px;
            vertical-align: top;
            font-size: 13px;
        }}
        th {{
            background: #e8f2ea;
            text-align: left;
            position: sticky;
            top: 0;
            z-index: 1;
        }}
        th span {{
            color: #666;
            font-weight: normal;
        }}
        .primary {{
            font-weight: bold;
        }}
        .daily-total td {{
            background: #f1f8f3;
            font-weight: bold;
        }}
        .muted {{
            color: #666;
            font-size: 12px;
            margin-top: 3px;
        }}
        .mono {{
            font-family: Menlo, Consolas, monospace;
        }}
        .scroll {{
            overflow-x: auto;
        }}
        .totals-strip table {{
            margin-top: 0;
        }}
        details summary {{
            cursor: pointer;
            font-weight: bold;
            color: #1b5e20;
        }}
        .nutrient-detail-row td {{
            background: #fafafa;
            padding: 6px 8px;
        }}
        .nested-table {{
            margin-top: 8px;
        }}
        .nested-table th {{
            position: static;
            background: #f0f0f0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>FNDDS Nutrient Results - {html_escape(record_id)}</h1>
        <div class="metadata">
            <span><strong>Subject:</strong> {html_escape(result_data.get('subject_id'))}</span>
            <span><strong>Date:</strong> {html_escape(result_data.get('date_of_record'))}</span>
            <span><strong>Day:</strong> {html_escape(result_data.get('day_of_week'))}</span>
            <span><strong>Diet:</strong> {html_escape(result_data.get('diet_type'))}</span>
            <span><strong>Reviewer:</strong> {html_escape(result_data.get('reviewer_id'))}</span>
            <div class="muted">{html_escape(result_data.get('nutrient_basis'))}</div>
        </div>

        <h2>Daily Summary</h2>
        <div class="totals-strip scroll">
            <table>
                <thead>
                    <tr>
                        <th>Meal</th>
                        {macro_header_cells}
                    </tr>
                </thead>
                <tbody>
                    {''.join(meal_summary_rows)}
                </tbody>
            </table>
        </div>

        <h2>Mapped Foods</h2>
        <div class="scroll">
            <table>
                <thead>
                    <tr>
                        <th>#</th>
                        <th>Time of Eat</th>
                        <th>Place of Eat</th>
                        <th>Meal</th>
                        <th>Mapped Food</th>
                        <th>Food Code</th>
                        <th>Original Amount</th>
                        <th>Grams</th>
                        <th>Mapped Portion</th>
                        <th>Conversion</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(rows)}
                </tbody>
            </table>
        </div>

        <h2>All Daily Nutrient Totals</h2>
        <div class="scroll">
            <table>
                <thead>
                    <tr>
                        <th>Nutrient Code</th>
                        <th>Tag</th>
                        <th>Description</th>
                        <th>Unit</th>
                        <th>Amount</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(all_total_rows)}
                </tbody>
            </table>
        </div>
    </div>
</body>
</html>
"""
        html_path = results_dir / f"{record_id}_fndds_nutrients.html"
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        html_files.append(html_path)
        print("✓")

    index_rows = []
    for html_file in html_files:
        json_file = html_file.with_suffix(".json")
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                result_data = json.load(f)
        except (OSError, json.JSONDecodeError):
            result_data = {}
        index_rows.append(f"""
            <tr>
                <td><a href="{html_escape(html_file.name)}">{html_escape(result_data.get('record_id') or html_file.stem)}</a></td>
                <td>{html_escape(result_data.get('subject_id'))}</td>
                <td>{html_escape(result_data.get('date_of_record'))}</td>
                <td>{html_escape(result_data.get('day_of_week'))}</td>
                <td>{html_escape(result_data.get('diet_type'))}</td>
                <td>{len(result_data.get('foods', []))}</td>
            </tr>
""")

    index_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8"/>
    <title>FNDDS Nutrient Results</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f7f7f7; color: #222; }}
        .container {{ max-width: 1000px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }}
        h1 {{ margin-top: 0; border-bottom: 3px solid #2e7d32; padding-bottom: 10px; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background: #e8f2ea; }}
        a {{ color: #1b5e20; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>FNDDS Nutrient Results</h1>
        <table>
            <thead>
                <tr>
                    <th>Record</th>
                    <th>Subject</th>
                    <th>Date</th>
                    <th>Day</th>
                    <th>Diet</th>
                    <th>Foods</th>
                </tr>
            </thead>
            <tbody>
                {''.join(index_rows)}
            </tbody>
        </table>
    </div>
</body>
</html>
"""
    index_path = results_dir / "index.html"
    with open(index_path, "w", encoding="utf-8") as f:
        f.write(index_html)
    html_files.append(index_path)

    print(f"\n✓ Generated {len(html_files)} FNDDS nutrient HTML files")
    return html_files


def main():
    """Main function to run the complete pipeline."""
    print("=" * 80)
    print("FNDDS Record Mapping Pipeline")
    print("=" * 80)
    
    # Step 1: Create fndds_food_index
    create_fndds_food_index(FNDDS_DB_PATH)
    
    # Step 2: Precompute FNDDS embeddings (sentence-transformers) and save to .npz
    if FNDDS_EMBEDDINGS_PATH.exists():
        reply = input("FNDDS embeddings file already exists. Regenerate? [y/N]: ").strip().lower()
        if reply in ("y", "yes"):
            build_fndds_embeddings(FNDDS_DB_PATH, FNDDS_EMBEDDINGS_PATH, EMBEDDING_MODEL_NAME)
        else:
            print("Using existing embeddings.")
    else:
        build_fndds_embeddings(FNDDS_DB_PATH, FNDDS_EMBEDDINGS_PATH, EMBEDDING_MODEL_NAME)
    
    # Step 3: Process records to JSON
    json_files = process_records_to_json(REVIEW_DIR, OUTPUT_DIR)
    
    if not json_files:
        print("\n⚠️  No records to process. Exiting.")
        return
    
    # Step 4: Normalize food descriptions with GPT-5.2
    normalized_files = normalize_records_with_gpt(json_files, OUTPUT_DIR)
    
    # Step 5: Match records to FNDDS (embedding similarity on normalized description, top 10)
    match_files = match_records_to_fndds(
        normalized_files,
        FNDDS_DB_PATH,
        OUTPUT_DIR,
        embeddings_path=FNDDS_EMBEDDINGS_PATH,
        embedding_model_name=EMBEDDING_MODEL_NAME,
    )
    
    # Step 6: Add portion-code mapping and gram conversion
    match_files = add_portion_mapping_to_matches(
        match_files,
        FNDDS_DB_PATH,
        OUTPUT_DIR,
    )
    
    # Step 7: Generate visualization HTML (top 10 per food)
    html_files = generate_visualization_html(match_files, OUTPUT_DIR, top_k=EMBEDDING_RERANK_TOP_K)
    
    # Step 8: Calculate FNDDS nutrient results
    result_files = generate_fndds_nutrient_results(
        match_files,
        FNDDS_DB_PATH,
        FNDDS_RESULTS_DIR,
    )
    
    # Step 9: Generate FNDDS nutrient result HTML reports
    result_html_files = generate_fndds_results_html(result_files, FNDDS_RESULTS_DIR)
    
    # Summary
    print("\n" + "=" * 80)
    print("PIPELINE SUMMARY")
    print("=" * 80)
    print(f"✓ Processed {len(json_files)} records")
    print(f"✓ Normalized {len(normalized_files)} records with GPT-5.2")
    print(f"✓ Generated {len(match_files)} match JSON files")
    print(f"✓ Generated {len(html_files)} visualization HTML files")
    print(f"✓ Generated {len(result_files)} FNDDS nutrient result files")
    print(f"✓ Generated {len(result_html_files)} FNDDS nutrient HTML files")
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"Results directory: {FNDDS_RESULTS_DIR}")
    print("\n✓ Pipeline completed successfully!")


if __name__ == "__main__":
    main()
