"""
Build a controlled dense retrieval FAISS index.

This script is intentionally separate from the locked runtime artifacts. It
encodes the existing product retrieval text with a supplied SentenceTransformer
model and writes an index in the same faiss.bin / ids.json layout used by
SearchAgent.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "data" / "product_base_retrieval" / "product_base_retrieval_text.json"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with input_path.open(encoding="utf-8") as f:
        rows = json.load(f)

    ids = []
    texts = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        pid = row.get("product_id")
        text = row.get("text") or ""
        if not pid or not text.strip():
            continue
        ids.append(str(pid))
        texts.append(text)

    print(f"[build_dense_retrieval_index] Loading model: {args.model_path}")
    model = SentenceTransformer(args.model_path)
    print(f"[build_dense_retrieval_index] Encoding {len(texts):,} products...")
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32")

    import faiss

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(np.ascontiguousarray(embeddings, dtype="float32"))

    faiss.write_index(index, str(output_dir / "faiss.bin"))
    with (output_dir / "ids.json").open("w", encoding="utf-8") as f:
        json.dump(ids, f)
    summary = {
        "model_path": str(args.model_path),
        "input": str(input_path),
        "output_dir": str(output_dir),
        "num_products": len(ids),
        "embedding_dim": int(embeddings.shape[1]),
    }
    with (output_dir / "build_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[build_dense_retrieval_index] Saved {output_dir}")


if __name__ == "__main__":
    main()
