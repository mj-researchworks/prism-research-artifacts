"""Build product-side embedding cache for the active MACPRS triplet reranker.

The autonomous/full evaluators use src.agents.reranker, whose default model is
e5_780_best_model and whose default cache path is product_embedding_cache.pt.
This script mirrors that reranker encoding path, so runtime reranking can reuse
product embeddings and only encode the query/context.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("HF_MODULES_CACHE", "/private/tmp/macprs_hf_modules_cache")

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = PROJECT_ROOT / "e5_780_best_model"
DEFAULT_PRODUCT_TEXT_JSON = PROJECT_ROOT / "data" / "product_base_retrieval" / "product_base_retrieval_text.json"
DEFAULT_OUT = PROJECT_ROOT / "product_embedding_cache.pt"
DEFAULT_QUERY_INSTRUCTION = (
    "Given a shopping request, retrieve products that satisfy "
    "the user's current requirements and preferences"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--product-text-json", type=Path, default=DEFAULT_PRODUCT_TEXT_JSON)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--limit", type=int, default=0, help="Optional smoke-test limit; 0 means all products.")
    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if not args.model_dir.exists():
        raise FileNotFoundError(f"Missing triplet model dir: {args.model_dir}")

    device = resolve_device(args.device)
    family_config = load_family_config(args.model_dir)
    is_qwen_decoder = is_qwen_decoder_triplet(family_config)

    print("Model dir:", args.model_dir)
    print("Product text JSON:", args.product_text_json)
    print("Output cache:", args.out)
    print("Device:", device)
    print("Batch size:", args.batch_size)
    print("Pooling:", "last_non_padding_token" if is_qwen_decoder else "mean")

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir), local_files_only=True, trust_remote_code=True)
    if is_qwen_decoder:
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

    model = AutoModel.from_pretrained(str(args.model_dir), local_files_only=True, trust_remote_code=True)
    model.config.use_cache = False
    model.to(device)
    model.eval()

    product_rows = load_product_texts(args.product_text_json)
    if args.limit > 0:
        product_rows = product_rows[: args.limit]
    embeddings = load_existing_embeddings(args.out) if args.resume else {}
    pending_rows = [(pid, text) for pid, text in product_rows if pid not in embeddings]

    print("Products total:", len(product_rows))
    print("Already cached:", len(embeddings))
    print("Remaining:", len(pending_rows))

    metadata = {
        "model_dir": str(args.model_dir),
        "product_text_json": str(args.product_text_json),
        "count": len(embeddings),
        "pooling": "last_non_padding_token" if is_qwen_decoder else "mean",
        "is_qwen_decoder": is_qwen_decoder,
        "query_instruction": family_config.get("query_instruction") or DEFAULT_QUERY_INSTRUCTION,
        "max_seq_length": args.max_length,
    }

    newly_encoded = 0
    for start in range(0, len(pending_rows), args.batch_size):
        batch_rows = pending_rows[start : start + args.batch_size]
        batch_ids = [pid for pid, _text in batch_rows]
        batch_texts = [text for _pid, text in batch_rows]
        batch_embeddings = encode_texts(
            model=model,
            tokenizer=tokenizer,
            texts=batch_texts,
            device=device,
            max_length=args.max_length,
            is_qwen_decoder=is_qwen_decoder,
        )
        for pid, emb in zip(batch_ids, batch_embeddings):
            embeddings[pid] = emb
        newly_encoded += len(batch_ids)

        total_done = len(embeddings)
        if args.save_every > 0 and newly_encoded % args.save_every == 0:
            metadata["count"] = len(embeddings)
            save_cache(args.out, metadata, embeddings)
            print(f"Checkpoint saved: {total_done}/{len(product_rows)} products", flush=True)

    metadata["count"] = len(embeddings)
    save_cache(args.out, metadata, embeddings)
    print("Saved product embedding cache:", args.out)
    print("Embedding count:", len(embeddings))


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_family_config(model_dir: Path) -> dict:
    for path in (model_dir / "macprs_triplet_family.json", model_dir.parent / "inference_config.json"):
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return {}
    return {}


def is_qwen_decoder_triplet(family_config: dict) -> bool:
    family = str(family_config.get("triplet_model_family") or "").lower()
    pooling = str(family_config.get("pooling") or "").lower()
    return "qwen" in family or pooling == "last_non_padding_token"


def load_product_texts(path: Path) -> list[tuple[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing product text JSON: {path}")
    rows = json.loads(path.read_text(encoding="utf-8"))
    product_texts: dict[str, str] = {}
    if isinstance(rows, dict):
        for pid, row in rows.items():
            if isinstance(row, dict):
                product_texts[str(row.get("product_id") or pid)] = str(row.get("text") or "")
            else:
                product_texts[str(pid)] = str(row or "")
    else:
        for row in rows:
            if isinstance(row, dict) and row.get("product_id"):
                product_texts[str(row["product_id"])] = str(row.get("text") or "")
    return [(pid, text) for pid, text in product_texts.items() if text.strip()]


def load_existing_embeddings(path: Path) -> dict:
    if not path.exists():
        return {}
    cache = torch.load(path, map_location="cpu")
    embeddings = cache.get("embeddings") if isinstance(cache, dict) else None
    if not isinstance(embeddings, dict):
        return {}
    return {
        str(pid): emb.detach().cpu() if hasattr(emb, "detach") else torch.as_tensor(emb)
        for pid, emb in embeddings.items()
    }


def save_cache(path: Path, metadata: dict, embeddings: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({**metadata, "count": len(embeddings), "embeddings": embeddings}, path)


def strip_legacy_prefix(text: str, prefix: str) -> str:
    text = str(text or "").strip()
    marker = prefix.lower() + ":"
    while text.lower().startswith(marker):
        text = text[len(marker) :].strip()
    return text


@torch.no_grad()
def encode_texts(model, tokenizer, texts: list[str], device: str, max_length: int, is_qwen_decoder: bool):
    formatted = [strip_legacy_prefix(text, "passage") for text in texts]
    batch = tokenizer(
        formatted,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    batch = {key: value.to(device) for key, value in batch.items()}
    output = model(**batch, use_cache=False)
    token_embeddings = output.last_hidden_state
    if is_qwen_decoder:
        pooled = last_token_pool(token_embeddings, batch["attention_mask"])
    else:
        mask = batch["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
        pooled = (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
    return F.normalize(pooled.float(), p=2, dim=1).cpu()


def last_token_pool(last_hidden_states, attention_mask):
    if bool((attention_mask[:, -1] == 1).all()):
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_indices = torch.arange(last_hidden_states.size(0), device=last_hidden_states.device)
    return last_hidden_states[batch_indices, sequence_lengths]


if __name__ == "__main__":
    main()
