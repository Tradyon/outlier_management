#!/usr/bin/env python3
"""
Semantic outlier detection using Gemini LLM.
"""

import argparse
import json
import logging
import os
import time

import pandas as pd
from tqdm import tqdm

try:
    from google import genai
except ImportError as exc:
    raise RuntimeError(
        "google-genai is not installed. Run: pip install google-genai"
    ) from exc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-3-flash-preview"
DEFAULT_CACHE = os.path.join("semantic_outlier", "semantic_outlier_llm_cache.jsonl")


def load_env_key():
    for key in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
        if key in os.environ and os.environ[key].strip():
            return os.environ[key].strip()
    env_path = ".env"
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                name, value = line.split("=", 1)
                name = name.strip()
                value = value.strip().strip('"').strip("'")
                if name in {"GOOGLE_API_KEY", "GEMINI_API_KEY"} and value:
                    return value
    return None


def normalize_hs_code(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    digits = "".join(ch for ch in text if ch.isdigit())
    if not digits:
        return None
    if len(digits) >= 6:
        return digits[:6].zfill(6)
    return digits.zfill(6)


def normalize_goods_shipped(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return text


def is_falsey(value):
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"false", "0", "f", "no", "n"}


def build_expected_description(record):
    description = (record.get("augmented_description") or "").strip()
    keywords = record.get("keywords") or []
    keywords_text = ", ".join(k.strip() for k in keywords if k and str(k).strip())
    if description and keywords_text:
        return f"{description} Keywords: {keywords_text}"
    if description:
        return description
    if keywords_text:
        return f"Keywords: {keywords_text}"
    return None


def load_hs_descriptions(jsonl_path):
    mapping = {}
    with open(jsonl_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            hs_code = str(record.get("id") or "").strip()
            if not hs_code:
                continue
            description = build_expected_description(record)
            if not description:
                continue
            mapping[hs_code] = description
    return mapping



def build_batch_prompt(items):
    lines = [
        "You are an HS-code classification assistant.",
        "Decide whether each shipment description clearly refers to a completely different product",
        "than the expected HS-6 description.",
        "Be conservative: only say different if it is obviously unrelated.",
        "If the shipment text is a bill of lading, random text, or too vague, return 'unclear'.",
        "",
        "Return JSON only as an array of objects with keys: row_id, verdict.",
        "verdict must be one of: same_product, different_product, unclear.",
        "Return one object per item using the same row_id.",
        "",
        "Items:",
    ]
    for item in items:
        lines.extend(
            [
                f"row_id: {item['row_id']}",
                f"goods_shipped: {item['goods_shipped']}",
                f"expected_hs_code: {item['hs_code']}",
                f"expected_description: {item['expected_description']}",
                "---",
            ]
        )
    return "\n".join(lines)


def call_gemini(client, model, prompt, response_schema):
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_json_schema": response_schema,
        },
    )
    return response.text


def load_cache(cache_path):
    cache = {}
    if not cache_path or not os.path.exists(cache_path):
        return cache
    with open(cache_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            key = record.get("key")
            if key:
                cache[key] = record.get("response")
    return cache


def append_cache(cache_path, key, response):
    if not cache_path:
        return
    record = {"key": key, "response": response}
    with open(cache_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def normalize_verdict(value):
    if value is None:
        return None
    verdict = str(value).strip().lower().replace(" ", "_")
    if verdict in {"same_product", "different_product", "unclear"}:
        return verdict
    return None


def verdict_from_cache(cached):
    if isinstance(cached, dict):
        return normalize_verdict(cached.get("verdict"))
    if isinstance(cached, str):
        return normalize_verdict(cached)
    return None


def parse_args():
    parser = argparse.ArgumentParser(description="Detect semantic outliers with Gemini.")
    parser.add_argument(
        "--input",
        default=os.path.join("semantic_outlier", "semantic_outlier_results.csv"),
        help="Input CSV containing goods_shipped and hs_code_6_digit",
    )
    parser.add_argument(
        "--hs-codes",
        default=os.path.join("semantic_outlier", "augmented_hs_codes.jsonl"),
        help="Path to augmented_hs_codes.jsonl",
    )
    parser.add_argument(
        "--output",
        default=os.path.join("semantic_outlier", "semantic_outlier_llm_results.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Gemini model name",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of rows for testing",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Sleep seconds between API calls",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Number of rows to group per LLM call",
    )
    parser.add_argument(
        "--cache",
        default=DEFAULT_CACHE,
        help="Cache file path for LLM responses",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    api_key = load_env_key()
    if not api_key:
        raise RuntimeError("Missing GOOGLE_API_KEY or GEMINI_API_KEY in env or .env")
    client = genai.Client(api_key=api_key)

    logger.info("Loading HS descriptions from %s", args.hs_codes)
    hs_descriptions = load_hs_descriptions(args.hs_codes)
    logger.info("Reading input CSV %s", args.input)
    df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    total_rows = len(df)
    if "is_multi_product_shipment" in df.columns:
        before = len(df)
        df = df[df["is_multi_product_shipment"].apply(is_falsey)].copy()
        logger.info("Filtered multi-product rows: %s -> %s", before, len(df))
    if args.limit:
        df = df.head(args.limit).copy()

    if "goods_shipped" not in df.columns:
        raise RuntimeError("Input CSV must include goods_shipped column")

    hs_col = "hs_code_6_digit" if "hs_code_6_digit" in df.columns else "hs_code"
    if hs_col not in df.columns:
        raise RuntimeError("Input CSV must include hs_code_6_digit or hs_code column")

    cache = load_cache(args.cache)
    logger.info("Loaded %s cached entries from %s", len(cache), args.cache)
    response_schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "row_id": {"type": "integer"},
                "verdict": {
                    "type": "string",
                    "enum": ["same_product", "different_product", "unclear"],
                },
            },
            "required": ["row_id", "verdict"],
        },
    }
    results = {}
    meta_by_row = {}
    pending = {}
    skipped_missing_goods = 0
    skipped_missing_hs = 0
    skipped_missing_desc = 0
    cache_hits = 0
    for row_id, row in enumerate(
        tqdm(df.itertuples(index=False), total=len(df), desc="Preparing rows"),
        start=1,
    ):
        goods_raw = getattr(row, "goods_shipped", None)
        goods_shipped = normalize_goods_shipped(goods_raw)
        shipment_id = getattr(row, "shipment_id", None)
        expected_hs_raw = getattr(row, hs_col, None)
        expected_hs = normalize_hs_code(expected_hs_raw)

        expected_desc = hs_descriptions.get(expected_hs) if expected_hs else None
        meta_by_row[row_id] = {
            "shipment_id": shipment_id,
            "goods_shipped": goods_raw,
            "hs_code_6_digit": expected_hs,
            "expected_description": expected_desc,
        }
        if not goods_shipped:
            skipped_missing_goods += 1
            results[row_id] = False
            continue
        if not expected_hs:
            skipped_missing_hs += 1
            results[row_id] = False
            continue
        if not expected_desc:
            skipped_missing_desc += 1
            results[row_id] = False
            continue

        key = f"{expected_hs}||{goods_shipped}"
        cached_verdict = verdict_from_cache(cache.get(key))
        if cached_verdict:
            cache_hits += 1
            results[row_id] = cached_verdict == "different_product"
            continue

        entry = pending.get(key)
        if not entry:
            entry = {
                "key": key,
                "row_id": row_id,
                "row_ids": [],
                "goods_shipped": goods_shipped,
                "hs_code": expected_hs,
                "expected_description": expected_desc,
            }
            pending[key] = entry
        entry["row_ids"].append(row_id)

    pending_items = list(pending.values())
    logger.info(
        "Rows total=%s, cache_hits=%s, missing_goods=%s, missing_hs=%s, missing_desc=%s",
        total_rows,
        cache_hits,
        skipped_missing_goods,
        skipped_missing_hs,
        skipped_missing_desc,
    )
    logger.info(
        "Prepared %s unique LLM inputs across %s rows", len(pending_items), len(df)
    )

    batch_iter = range(0, len(pending_items), args.batch_size)
    for i in tqdm(batch_iter, desc="LLM batches"):
        batch = pending_items[i : i + args.batch_size]
        prompt = build_batch_prompt(batch)
        raw_text = call_gemini(client, args.model, prompt, response_schema)
        parsed = []
        if raw_text:
            try:
                parsed = json.loads(raw_text)
            except json.JSONDecodeError:
                parsed = []
        verdict_map = {}
        if isinstance(parsed, list):
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                row_id = item.get("row_id")
                verdict = normalize_verdict(item.get("verdict"))
                if verdict is None:
                    verdict = "unclear"
                try:
                    row_id = int(row_id)
                except (TypeError, ValueError):
                    continue
                verdict_map[row_id] = verdict

        for item in batch:
            verdict = verdict_map.get(item["row_id"], "unclear")
            cache[item["key"]] = {"verdict": verdict}
            append_cache(args.cache, item["key"], {"verdict": verdict})
            for row_id in item["row_ids"]:
                results[row_id] = verdict == "different_product"

        if args.sleep:
            time.sleep(args.sleep)

    outliers = []
    for row_id in range(1, len(df) + 1):
        if not results.get(row_id, False):
            continue
        meta = meta_by_row.get(row_id, {})
        outliers.append(
            {
                "shipment_id": meta.get("shipment_id"),
                "hs_code_6_digit": meta.get("hs_code_6_digit"),
                "goods_shipped": meta.get("goods_shipped"),
                "expected_description": meta.get("expected_description"),
            }
        )
    out_df = pd.DataFrame(outliers)
    out_df.to_csv(args.output, index=False)
    logger.info("Saved outliers to %s", args.output)
    logger.info("Semantic outliers: %s / %s", len(out_df), len(df))


if __name__ == "__main__":
    main()
