"""
Structura AI - Core AI Extraction Logic using OpenAI
"""
import os
import json
import re
import time
import hashlib
from typing import Any, Dict, Optional, List
from openai import OpenAI

# Initialize OpenAI client
client = None


def get_openai_client() -> OpenAI:
    """Lazy-load OpenAI client."""
    global client
    if client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable is required")
        client = OpenAI(api_key=api_key)
    return client


# --- System Prompts ---

EXTRACTION_SYSTEM_PROMPT = """You are Structura, a precision data extraction engine.

ROLE: Extract structured data from unstructured text according to a user-defined JSON schema.

RULES:
1. ONLY extract information explicitly present in or strongly implied by the input text.
2. NEVER fabricate, hallucinate, or infer data that is not supported by the text.
3. For fields where information is not available, return null (unless the field is required).
4. Respect the schema type constraints exactly:
   - "string" -> return a string
   - "integer" -> return a whole number
   - "number" -> return a decimal
   - "boolean" -> return true/false
   - "array" -> return an array
   - "enum" -> return ONLY one of the allowed values
5. When a "description" is provided for a field, use it as guidance for what to extract.
6. For scoring/rating fields with min/max, use the full range based on available signals.
7. Return ONLY valid JSON matching the provided schema. No markdown, no explanations."""

CLASSIFICATION_SYSTEM_PROMPT = """You are Structura Classifier, a precise text classification engine.

ROLE: Classify the given text into one or more categories from the provided list.

RULES:
1. Choose the MOST relevant category based on the primary intent/topic of the text.
2. If multi_label is enabled, return all applicable categories.
3. Include a confidence score (0.0-1.0) for each classification.
4. If include_reasoning is true, add a brief one-sentence explanation.
5. Only use categories from the provided list. Never create new categories.
6. Return valid JSON with the structure: {"primary": {"category": "...", "confidence": 0.0-1.0, "reasoning": "..."}, "secondary": [{"category": "...", "confidence": 0.0-1.0}]}"""

TRANSFORM_SYSTEM_PROMPT = """You are Structura Transformer, a data normalization and enrichment engine.

ROLE: Transform messy or semi-structured input data into clean, properly formatted output matching the target schema.

RULES:
1. Normalize names (proper capitalization), addresses (standard format), phones (requested format).
2. Split combined fields into structured sub-fields when the target schema requires it.
3. Extract implicit information (e.g., dietary restrictions from a notes field).
4. Do NOT add information that is not present or implied in the input.
5. For fields that cannot be determined from input, return null.
6. Return valid JSON matching the target schema exactly."""


# --- Model Mapping ---

MODEL_MAP = {
    "fast": "gpt-4o-mini",
    "quality": "gpt-4o",
}


# --- In-Memory Cache (MVP) ---

_cache: Dict[str, dict] = {}
MAX_CACHE_SIZE = 1000


def _cache_key(text: str, schema: dict, endpoint: str) -> str:
    """Generate a cache key from text + schema."""
    content = json.dumps({"text": text, "schema": schema, "endpoint": endpoint}, sort_keys=True)
    return hashlib.md5(content.encode()).hexdigest()


def _get_cached(cache_key: str) -> Optional[dict]:
    """Get cached result if available."""
    if cache_key in _cache:
        return _cache[cache_key]
    return None


def _set_cache(cache_key: str, result: dict):
    """Cache a result."""
    if len(_cache) >= MAX_CACHE_SIZE:
        # Evict oldest entry
        oldest_key = next(iter(_cache))
        del _cache[oldest_key]
    _cache[cache_key] = result


# --- Core Extraction ---

def extract_data(text: str, schema: dict, model: str = "fast",
                 confidence_scores: bool = False, strict_mode: bool = True) -> dict:
    """
    Extract structured data from unstructured text using OpenAI.
    
    Returns: {data, confidence, validation, usage, cached}
    """
    start_time = time.time()

    # Check cache
    cache_key = _cache_key(text, schema, "extract")
    cached = _get_cached(cache_key)
    if cached:
        cached["cached"] = True
        return cached

    openai_model = MODEL_MAP.get(model, "gpt-4o-mini")
    client = get_openai_client()

    # Build prompt
    user_prompt = f"""SCHEMA TO EXTRACT:
{json.dumps(schema, indent=2)}

TEXT TO EXTRACT FROM:
\"\"\"{text}\"\"\"

Extract all matching fields from the text above. Return valid JSON only."""

    try:
        response = client.chat.completions.create(
            model=openai_model,
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )

        result_text = response.choices[0].message.content
        extracted_data = json.loads(result_text)

        # Calculate usage
        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0
        latency_ms = int((time.time() - start_time) * 1000)

        # Validate against schema
        validation = validate_extraction(extracted_data, schema, strict_mode)

        # Calculate confidence scores if requested
        confidence = None
        if confidence_scores:
            confidence = calculate_confidence_scores(text, extracted_data, schema)

        result = {
            "data": extracted_data,
            "confidence": confidence,
            "validation": validation,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "model": model,
                "cost_credits": 1.0 if model == "fast" else 3.0,
            },
            "latency_ms": latency_ms,
            "cached": False,
        }

        # Cache the result
        _set_cache(cache_key, result)

        return result

    except json.JSONDecodeError as e:
        raise ValueError(f"AI returned invalid JSON: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Extraction failed: {str(e)}")


# --- Classification ---

def classify_text(text: str, categories: list, multi_label: bool = False,
                  include_reasoning: bool = False) -> dict:
    """
    Classify text into provided categories.
    
    Returns: {classification, secondary_categories, usage}
    """
    start_time = time.time()
    client = get_openai_client()

    categories_text = "\n".join([
        f"- {cat['name']}: {cat['description']}" for cat in categories
    ])

    user_prompt = f"""CATEGORIES:
{categories_text}

TEXT TO CLASSIFY:
\"\"\"{text}\"\"\"

Multi-label: {multi_label}
Include reasoning: {include_reasoning}

Classify the text. Return JSON with: {{"primary": {{"category": "name", "confidence": 0.0-1.0, "reasoning": "...or null"}}, "secondary": [{{"category": "name", "confidence": 0.0-1.0}}]}}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": CLASSIFICATION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )

        result_text = response.choices[0].message.content
        result_data = json.loads(result_text)
        latency_ms = int((time.time() - start_time) * 1000)

        primary = result_data.get("primary", {})
        secondary = result_data.get("secondary", [])

        classification = {
            "category": primary.get("category", categories[0]["name"]),
            "confidence": primary.get("confidence", 0.5),
            "reasoning": primary.get("reasoning") if include_reasoning else None,
        }

        secondary_categories = [
            {"category": s.get("category"), "confidence": s.get("confidence", 0.0)}
            for s in secondary if s.get("category") != classification["category"]
        ]

        return {
            "classification": classification,
            "secondary_categories": secondary_categories,
            "usage": {"cost_credits": 0.5},
            "latency_ms": latency_ms,
        }

    except Exception as e:
        raise RuntimeError(f"Classification failed: {str(e)}")


# --- Transform ---

def transform_data(input_data: dict, target_schema: dict) -> dict:
    """
    Validate and transform messy structured data.
    
    Returns: {data, usage}
    """
    start_time = time.time()
    client = get_openai_client()

    user_prompt = f"""INPUT DATA:
{json.dumps(input_data, indent=2)}

TARGET SCHEMA:
{json.dumps(target_schema, indent=2)}

Transform the input data to match the target schema. Normalize, clean, and restructure as needed. Return valid JSON only."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": TRANSFORM_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )

        result_text = response.choices[0].message.content
        transformed_data = json.loads(result_text)
        latency_ms = int((time.time() - start_time) * 1000)

        return {
            "data": transformed_data,
            "usage": {"cost_credits": 1.0},
            "latency_ms": latency_ms,
        }

    except Exception as e:
        raise RuntimeError(f"Transform failed: {str(e)}")


# --- Validation ---

def validate_extraction(result: dict, schema: dict, strict: bool = True) -> dict:
    """Validate extracted data against the original schema."""
    errors = []
    warnings = []

    # Check required fields
    required = schema.get("required", [])
    for field in required:
        if field not in result or result[field] is None:
            if strict:
                errors.append(f"Required field '{field}' is missing")
            else:
                warnings.append(f"Required field '{field}' could not be extracted")

    # Check types
    properties = schema.get("properties", {})
    for field, value in result.items():
        if field in properties and value is not None:
            expected_type = properties[field].get("type")
            if not _type_matches(value, expected_type):
                errors.append(
                    f"Field '{field}' expected {expected_type}, got {type(value).__name__}"
                )

    # Check enums
    for field, spec in properties.items():
        if "enum" in spec and field in result and result[field] is not None:
            if result[field] not in spec["enum"]:
                errors.append(
                    f"Field '{field}' value '{result[field]}' not in allowed: {spec['enum']}"
                )

    # Check min/max
    for field, spec in properties.items():
        if field in result and result[field] is not None:
            if "minimum" in spec and isinstance(result[field], (int, float)):
                if result[field] < spec["minimum"]:
                    errors.append(f"Field '{field}' below minimum ({spec['minimum']})")
            if "maximum" in spec and isinstance(result[field], (int, float)):
                if result[field] > spec["maximum"]:
                    errors.append(f"Field '{field}' above maximum ({spec['maximum']})")

    return {
        "all_required_present": len([f for f in required if f in result and result[f] is not None]) == len(required),
        "type_errors": errors,
        "warnings": warnings,
    }


def _type_matches(value: Any, expected_type: str) -> bool:
    """Check if a value matches the expected JSON schema type."""
    if expected_type is None:
        return True
    type_map = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "array": list,
        "object": dict,
    }
    expected = type_map.get(expected_type)
    if expected is None:
        return True
    return isinstance(value, expected)


# --- Confidence Scores ---

def calculate_confidence_scores(text: str, extracted: dict, schema: dict) -> dict:
    """
    Heuristic confidence scoring for each extracted field.
    """
    confidence = {}
    properties = schema.get("properties", {})
    text_lower = text.lower()

    for field, value in extracted.items():
        if value is None:
            confidence[field] = 0.0
            continue

        score = 0.7  # base confidence

        # Boost: value appears literally in text
        str_value = str(value).lower()
        if str_value and str_value in text_lower:
            score += 0.2

        # Boost: structured format (email, phone, URL)
        if isinstance(value, str):
            if re.match(r'.+@.+\..+', value):
                score += 0.1
            elif re.match(r'[\d\-\(\)\+\s]{7,}', value):
                score += 0.08

        # Penalty: very long generated text
        if isinstance(value, str) and len(value) > 200:
            score -= 0.1

        # Boost: enum field with matching value
        if field in properties:
            spec = properties[field]
            if "enum" in spec and value in spec["enum"]:
                score += 0.05

        confidence[field] = round(min(max(score, 0.0), 1.0), 2)

    return confidence
