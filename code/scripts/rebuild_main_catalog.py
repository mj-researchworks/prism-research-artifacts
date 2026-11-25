#!/usr/bin/env python3
"""Rebuild PRISM's frozen product table from the pinned English PSCon graph."""

import argparse
import hashlib
import json
from pathlib import Path

GRAPH_SHA256 = "d7b9dacc8c83aacaa174bb9955ec4240531076f1ff155bc4ccd07d4e53a4293b"
TABLE_SHA256 = "59aaf35f9fa32ec84a0d91190571b0b0caf3947801cba1b651692bc12271b26d"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("graph", type=Path, help="Pinned PSCon knowledgeGraph_en.txt")
    parser.add_argument("output", type=Path, help="Destination for product_base_table.json")
    args = parser.parse_args()

    source = args.graph.read_bytes()
    if hashlib.sha256(source).hexdigest() != GRAPH_SHA256:
        parser.error("Input checksum differs from the pinned English PSCon graph.")

    products = {}
    for line in source.decode("utf-8").split("\n"):
        parts = line.split(", ", 2)
        if len(parts) != 3:
            continue
        product_id, key, value = parts
        key = " ".join(key.split())
        value = " ".join(value.split())
        attributes = products.setdefault(product_id, {})
        if key and value and key not in {"like", "dislike"}:
            attributes[key] = value

    table = {}
    for product_id, attributes in products.items():
        if not attributes:
            continue
        title = attributes.pop("title", "")
        lower_keys = {key.lower(): value for key, value in attributes.items()}
        table[product_id] = {
            "product_id": product_id,
            "title": title,
            "brand": lower_keys.get("brand", ""),
            "color": lower_keys.get("color", ""),
            "attributes": attributes,
        }

    output = json.dumps(table, ensure_ascii=False, indent=2).encode("utf-8")
    if len(table) != 22203 or hashlib.sha256(output).hexdigest() != TABLE_SHA256:
        parser.exit(1, "Rebuilt table differs from the recorded PRISM snapshot.\n")

    args.output.write_bytes(output)
    print(f"Wrote {len(table):,} products to {args.output}")
    print(f"SHA-256: {TABLE_SHA256}")


if __name__ == "__main__":
    main()
