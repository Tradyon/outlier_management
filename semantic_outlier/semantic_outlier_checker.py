#!/usr/bin/env python3
"""
Semantic outlier detection for shipment_master.csv using Weaviate.
"""

import argparse
import json
import logging
import math
import os
import time
import uuid
import urllib.error
import urllib.request

import pandas as pd

CLASS_NAME = "HsCode"
NOISE_TOKENS = {
    "item",
    "factura",
    "invoice",
    "toneladas",
    "tonelada",
    "tonnes",
    "tons",
    "ton",
    "mt",
    "mts",
    "kg",
    "kgs",
    "g",
    "lb",
    "lbs",
    "load",
    "count",
    "weight",
    "net",
    "gross",
    "bags",
    "bag",
    "sacks",
    "sack",
    "pallet",
    "pallets",
    "carton",
    "cartons",
    "box",
    "boxes",
    "shipment",
    "shipper",
    "consignee",
    "bill",
    "lading",
    "bl",
    "po",
    "ref",
    "reference",
    "lot",
    "lots",
    "container",
    "containers",
    "the",
    "and",
    "of",
    "for",
    "para",
    "de",
    "la",
    "el",
    "y",
    "en",
    "con",
}
MAX_QUERY_TOKENS = 12

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def http_json(method, url, payload=None, timeout=30):
    data = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            if not body:
                return {}
            return json.loads(body)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8")
        raise RuntimeError(f"HTTP {exc.code} {exc.reason} for {url}: {body}") from exc


def wait_for_weaviate(base_url, timeout=120):
    start = time.time()
    while True:
        try:
            http_json("GET", f"{base_url}/v1/.well-known/ready", timeout=5)
            return
        except Exception as exc:
            if time.time() - start > timeout:
                raise RuntimeError("Weaviate did not become ready in time.") from exc
            time.sleep(2)


def class_exists(schema, class_name):
    for entry in schema.get("classes", []):
        if entry.get("class") == class_name:
            return True
    return False


def ensure_schema(base_url, force_recreate=False):
    schema = http_json("GET", f"{base_url}/v1/schema")
    exists = class_exists(schema, CLASS_NAME)
    if exists and force_recreate:
        logger.info("Dropping existing schema for %s", CLASS_NAME)
        http_json("DELETE", f"{base_url}/v1/schema/{CLASS_NAME}")
        exists = False
    if not exists:
        logger.info("Creating schema for %s", CLASS_NAME)
        payload = {
            "class": CLASS_NAME,
            "vectorizer": "text2vec-transformers",
            "moduleConfig": {
                "text2vec-transformers": {
                    "vectorizeClassName": False
                }
            },
            "properties": [
                {"name": "hs_code", "dataType": ["text"]},
                {"name": "content", "dataType": ["text"]},
            ],
        }
        http_json("POST", f"{base_url}/v1/schema", payload)


def graphql_query(base_url, query):
    payload = {"query": query}
    response = http_json("POST", f"{base_url}/v1/graphql", payload)
    if "errors" in response:
        raise RuntimeError(response["errors"])
    return response.get("data", {})


def get_class_count(base_url):
    query = f"{{ Aggregate {{ {CLASS_NAME} {{ meta {{ count }} }} }} }}"
    data = graphql_query(base_url, query)
    aggregate = data.get("Aggregate", {}).get(CLASS_NAME, [])
    if not aggregate:
        return 0
    return int(aggregate[0].get("meta", {}).get("count", 0))


def build_content(record):
    description = (record.get("augmented_description") or "").strip()
    keywords = record.get("keywords") or []
    keywords_text = ", ".join(k.strip() for k in keywords if k and str(k).strip())
    if description and keywords_text:
        return f"{description} Keywords: {keywords_text}"
    if description:
        return description
    if keywords_text:
        return f"Keywords: {keywords_text}"
    return ""


def iter_hs_codes(jsonl_path):
    with open(jsonl_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            hs_code = str(record.get("id") or "").strip()
            if not hs_code:
                continue
            content = build_content(record)
            if not content:
                continue
            yield hs_code, content


def ingest_hs_codes(base_url, jsonl_path, batch_size=128):
    objects = []
    total = 0
    for hs_code, content in iter_hs_codes(jsonl_path):
        object_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"hs_code:{hs_code}"))
        objects.append(
            {
                "class": CLASS_NAME,
                "id": object_id,
                "properties": {
                    "hs_code": hs_code,
                    "content": content,
                },
            }
        )
        if len(objects) >= batch_size:
            post_batch(base_url, objects)
            total += len(objects)
            objects = []
    if objects:
        post_batch(base_url, objects)
        total += len(objects)
    return total


def post_batch(base_url, objects):
    response = http_json("POST", f"{base_url}/v1/batch/objects", {"objects": objects})
    if isinstance(response, list):
        failed = [
            item
            for item in response
            if isinstance(item, dict)
            and item.get("result", {}).get("errors")
        ]
        if failed:
            logger.warning("Batch insert returned %s errors", len(failed))
    elif isinstance(response, dict):
        errors = response.get("errors")
        if errors:
            logger.warning("Batch insert returned errors: %s", errors)


def normalize_goods_shipped(value):
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return text


def clean_goods_shipped(text, max_tokens=MAX_QUERY_TOKENS):
    if not text:
        return None
    lowered = text.lower()
    cleaned = []
    for ch in lowered:
        if ch.isalpha() or ch.isspace():
            cleaned.append(ch)
        else:
            cleaned.append(" ")
    tokens = [token for token in "".join(cleaned).split() if token not in NOISE_TOKENS]
    if not tokens:
        return None
    if max_tokens:
        tokens = tokens[:max_tokens]
    return " ".join(tokens)


def normalize_hs_code(value):
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
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


def get_expected_hs_code(row):
    code = normalize_hs_code(row.get("hs_code_6_digit"))
    if code:
        return code
    return normalize_hs_code(row.get("hs_code"))


def is_falsey(value):
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    text = str(value).strip().lower()
    return text in {"false", "0", "f", "no", "n"}


def graphql_string(value):
    return json.dumps(value)


def parse_matches(results, use_rerank=True):
    matches = []
    for item in results:
        additional = item.get("_additional", {})
        rerank_score = None
        if use_rerank:
            rerank = additional.get("rerank")
            if isinstance(rerank, list) and rerank:
                rerank_score = rerank[0].get("score")
            elif isinstance(rerank, dict):
                rerank_score = rerank.get("score")
        matches.append(
            {
                "hs_code": item.get("hs_code"),
                "rerank_score": rerank_score,
                "distance": additional.get("distance"),
            }
        )
    if use_rerank and matches:
        matches.sort(
            key=lambda entry: (entry["rerank_score"] is not None, entry["rerank_score"]),
            reverse=True,
        )
    return matches


def query_top_matches(base_url, query_text, limit=5, use_rerank=True):
    if use_rerank:
        additional_clause = (
            "distance rerank("
            f"property: \"content\", query: {graphql_string(query_text)}"
            ") { score }"
        )
    else:
        additional_clause = "distance"
    query = (
        f"{{ Get {{ {CLASS_NAME}("
        f"nearText: {{ concepts: [{graphql_string(query_text)}] }}"
        f" limit: {limit}"
        f") {{ hs_code _additional {{ {additional_clause} }} }} }} }}"
    )
    data = graphql_query(base_url, query)
    results = data.get("Get", {}).get(CLASS_NAME, [])
    if not results:
        return []
    return parse_matches(results, use_rerank=use_rerank)


def query_with_fallback(base_url, query_text, limit=5):
    try:
        return query_top_matches(base_url, query_text, limit=limit, use_rerank=True)
    except RuntimeError as exc:
        logger.warning("Rerank query failed, falling back to vector search: %s", exc)
        return query_top_matches(base_url, query_text, limit=limit, use_rerank=False)


def load_shipments(csv_path, limit=None):
    usecols = [
        "shipment_id",
        "goods_shipped",
        "hs_code",
        "hs_code_6_digit",
        "is_multi_product_shipment",
    ]
    df = pd.read_csv(csv_path, usecols=usecols, dtype=str, keep_default_na=False)
    df["is_multi_product_shipment"] = df["is_multi_product_shipment"].astype(str)
    filtered = df[df["is_multi_product_shipment"].apply(is_falsey)].copy()
    if limit:
        filtered = filtered.head(limit).copy()
    filtered["goods_shipped_clean"] = filtered["goods_shipped"].apply(normalize_goods_shipped)
    filtered["goods_shipped_query"] = filtered["goods_shipped_clean"].apply(clean_goods_shipped)
    filtered["goods_shipped_query"] = filtered["goods_shipped_query"].where(
        filtered["goods_shipped_query"].notna(), filtered["goods_shipped_clean"]
    )
    filtered["expected_hs_code"] = filtered.apply(get_expected_hs_code, axis=1)
    return filtered


def build_match_cache(base_url, goods_shipped_values, top_k):
    cache = {}
    for idx, goods in enumerate(goods_shipped_values, start=1):
        matches = query_with_fallback(base_url, goods, limit=top_k)
        cache[goods] = matches
        if idx % 100 == 0:
            logger.info("Queried %s/%s goods descriptions", idx, len(goods_shipped_values))
    return cache


def annotate_outliers(df, match_cache, match_top_k=1, hs_code_set=None):
    rows = []
    for row in df.itertuples(index=False):
        goods = row.goods_shipped_clean
        query_text = row.goods_shipped_query or goods
        expected = row.expected_hs_code
        predicted = None
        rerank_score = None
        distance = None
        match_rank = None
        top_hs_codes = []
        reason = None
        is_outlier = True

        if not goods:
            reason = "missing_goods_shipped"
        elif not expected:
            reason = "missing_hs_code"
        elif hs_code_set is not None and expected not in hs_code_set:
            reason = "expected_not_in_index"
            is_outlier = None
        else:
            matches = match_cache.get(query_text, [])
            if matches:
                top_hs_codes = [m.get("hs_code") for m in matches if m.get("hs_code")]
                best = matches[0]
                predicted = best.get("hs_code")
                rerank_score = best.get("rerank_score")
                distance = best.get("distance")
                if expected in top_hs_codes:
                    match_rank = top_hs_codes.index(expected) + 1
            if not matches:
                reason = "no_match"
            elif match_rank is not None and match_rank <= match_top_k:
                is_outlier = False
                reason = "match"
            else:
                if hs_code_set is not None and expected not in hs_code_set:
                    reason = "expected_not_in_index"
                else:
                    reason = "hs_code_mismatch"

        rows.append(
            {
                "shipment_id": row.shipment_id,
                "goods_shipped": row.goods_shipped,
                "goods_shipped_query": query_text,
                "hs_code_6_digit": expected,
                "predicted_hs_code": predicted,
                "rerank_score": rerank_score,
                "distance": distance,
                "match_rank": match_rank,
                "top_hs_codes": json.dumps(top_hs_codes),
                "expected_in_index": (
                    None if expected is None or hs_code_set is None else expected in hs_code_set
                ),
                "is_semantic_outlier": is_outlier,
                "semantic_outlier_reason": reason,
            }
        )
    return pd.DataFrame(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Detect semantic outliers with Weaviate.")
    parser.add_argument(
        "--shipments",
        default="shipment_master.csv",
        help="Path to shipment_master.csv",
    )
    parser.add_argument(
        "--hs-codes",
        default=os.path.join("semantic_outlier", "augmented_hs_codes.jsonl"),
        help="Path to augmented_hs_codes.jsonl",
    )
    parser.add_argument(
        "--weaviate-url",
        default="http://localhost:8080",
        help="Weaviate base URL",
    )
    parser.add_argument(
        "--output",
        default=os.path.join("semantic_outlier", "semantic_outlier_results.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of shipment rows for testing",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top K candidates to retrieve from Weaviate",
    )
    parser.add_argument(
        "--match-top-k",
        type=int,
        default=5,
        help="Treat matches within top K candidates as non-outliers",
    )
    parser.add_argument(
        "--force-reindex",
        action="store_true",
        help="Drop and recreate Weaviate schema before ingest",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    wait_for_weaviate(args.weaviate_url)
    ensure_schema(args.weaviate_url, force_recreate=args.force_reindex)

    count = get_class_count(args.weaviate_url)
    if count == 0 or args.force_reindex:
        logger.info("Indexing HS codes from %s", args.hs_codes)
        total = ingest_hs_codes(args.weaviate_url, args.hs_codes)
        logger.info("Indexed %s HS code entries", total)
    else:
        logger.info("HS code index already populated (%s objects). Skipping ingest.", count)

    hs_code_set = {hs for hs, _ in iter_hs_codes(args.hs_codes)}

    df = load_shipments(args.shipments, limit=args.limit)
    logger.info("Loaded %s shipment rows for semantic analysis", len(df))

    if args.match_top_k > args.top_k:
        logger.warning(
            "--match-top-k=%s is greater than --top-k=%s; clamping to %s",
            args.match_top_k,
            args.top_k,
            args.top_k,
        )
        args.match_top_k = args.top_k

    if hs_code_set:
        eligible = df[df["expected_hs_code"].isin(hs_code_set)]
    else:
        eligible = df
    unique_goods = sorted(set(eligible["goods_shipped_query"].dropna()))
    logger.info(
        "Querying Weaviate for %s unique goods descriptions", len(unique_goods)
    )
    match_cache = build_match_cache(args.weaviate_url, unique_goods, args.top_k)

    out_df = annotate_outliers(
        df,
        match_cache,
        match_top_k=args.match_top_k,
        hs_code_set=hs_code_set,
    )
    out_df.to_csv(args.output, index=False)

    outlier_count = int(out_df["is_semantic_outlier"].sum())
    logger.info("Saved results to %s", args.output)
    logger.info("Semantic outliers: %s / %s", outlier_count, len(out_df))


if __name__ == "__main__":
    main()
