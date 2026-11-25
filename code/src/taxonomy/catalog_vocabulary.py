"""
Catalog vocabulary helpers for retrieval query expansion.

This is intentionally lightweight and deterministic: it mines product-type
labels from the local FAISS index folders and brands from catalog_brands.json,
then returns short recall-only query variants. It does not route, filter, or
change taxonomy scope.
"""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
INDEXES_PATH = ROOT / "indexes"
BRANDS_PATH = ROOT / "src" / "taxonomy" / "catalog_brands.json"

GENERIC_TOKENS = {
    "and", "or", "the", "for", "with", "from", "home", "general",
    "products", "product", "items", "accessories", "supplies", "living",
    "electronics", "fashion", "clothing", "office", "stationery",
}

BAD_SINGLE_TOKEN_ALIASES = {
    "set", "sets", "case", "cases", "basic", "general", "portable",
    "traditional", "single", "standard", "mixed", "all", "one",
}

BAD_CATEGORY_QUERY_LABELS = {
    "cell phone tripods",
    "cell phone basic cases",
    "cell phone screen protectors",
    "screen protectors",
    "smartwatches",
    "computer monitors",
    "feather dusters",
    "playing card decks",
    "baseball bats",
    "cutting boards",
    "wall clocks",
    "coffee machines",
    "disposable cups",
    "portable and arranger keyboards",
    "qled tvs",
}

PREFIX_TOKENS = {
    "pc", "mac", "cell", "phone", "home", "office", "kids", "kid",
    "men", "mens", "women", "womens", "adult", "portable", "traditional",
    "standard", "single", "wireless",
}

HEAD_SAFE_TOKENS = {
    "phone", "phones", "laptop", "laptops", "computer", "computers",
    "keyboard", "keyboards", "mouse", "mice", "headphone", "headphones",
    "headset", "headsets", "earbuds", "tablet", "tablets", "monitor",
    "monitors", "charger", "chargers", "watch", "watches", "tv", "tvs",
    "camera", "cameras", "camcorder", "camcorders", "router", "routers",
    "bag", "bags", "backpack", "backpacks", "wallet", "wallets", "shoe",
    "shoes", "boot", "boots", "jacket", "jackets", "coat", "coats",
    "shirt", "shirts", "dress", "dresses", "skirt", "skirts", "pants",
    "jeans", "socks", "stockings", "luggage", "suitcase", "suitcases",
    "food", "treats", "snacks", "coffee", "tea", "drink", "drinks",
    "candy", "spices", "flowers", "plant", "plants", "lamp", "lamps",
    "desk", "desks", "chair", "chairs", "mirror", "mirrors", "clock",
    "clocks", "mug", "mugs", "glass", "glasses", "bottle", "bottles",
    "cup", "cups", "kettle", "kettles", "refrigerator", "refrigerators",
    "washer", "washers", "machine", "machines", "cooker", "cookers",
    "board", "boards", "mat", "mats", "lock", "locks", "piano", "pianos",
    "toy", "toys", "car", "cars", "doll", "dolls", "bicycle", "bicycles",
    "book", "books", "notebook", "notebooks", "pen", "pens", "paper",
}

IRREGULAR = {
    "mice": "mouse",
    "knives": "knife",
    "glasses": "glass",
    "watches": "watch",
    "dresses": "dress",
}


def _norm(text: str) -> str:
    text = str(text or "").lower().replace("&", " and ")
    text = re.sub(r"[^a-z0-9+\-\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _singular(token: str) -> str:
    if token in IRREGULAR:
        return IRREGULAR[token]
    if token.endswith("cases"):
        return token[:-1]
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith(("ches", "shes", "xes", "ses")):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _tokens(text: str) -> list[str]:
    return [
        _singular(t)
        for t in re.findall(r"[a-z0-9]+", _norm(text))
        if t not in GENERIC_TOKENS and len(t) > 1
    ]


def _clean_alias(alias: str) -> str:
    tokens = _tokens(alias)
    if not tokens:
        return ""
    if len(tokens) == 1 and (tokens[0] in BAD_SINGLE_TOKEN_ALIASES or len(tokens[0]) < 4):
        return ""
    return " ".join(tokens)


def _label_aliases(label: str) -> set[str]:
    tokens = _tokens(label)
    aliases = set()
    if not tokens:
        return aliases

    aliases.add(" ".join(tokens))

    # Remove catalog/platform prefixes while preserving the product head.
    trimmed = list(tokens)
    while len(trimmed) > 1 and trimmed[0] in PREFIX_TOKENS:
        trimmed = trimmed[1:]
        aliases.add(" ".join(trimmed))

    if len(tokens) >= 2:
        aliases.add(" ".join(tokens[-2:]))
    if tokens[-1] in HEAD_SAFE_TOKENS:
        aliases.add(tokens[-1])

    # Product-specific aliases that recur in this catalog.
    joined = " ".join(tokens)
    if "cell phone" in joined:
        aliases.update({"phone", "mobile phone", "smartphone", joined})
    if "laptop" in joined or "notebook" in joined:
        aliases.update({"laptop", "laptop computer", joined})
    if "headphone" in joined or "headset" in joined or "earbud" in joined:
        aliases.update({"headphones", "headset", joined})
    if "tv" in joined or "television" in joined:
        aliases.update({"tv", "television", joined})
    if "mice" in _norm(label) or "mouse" in joined:
        aliases.update({"mouse", "computer mouse", joined})

    return {_clean_alias(a) for a in aliases if _clean_alias(a)}


@lru_cache(maxsize=1)
def _catalog_records() -> tuple[tuple[dict, ...], tuple[str, ...]]:
    records = []
    if INDEXES_PATH.exists():
        for l1_dir in INDEXES_PATH.iterdir():
            if not l1_dir.is_dir() or l1_dir.name.startswith("_"):
                continue
            for l2_dir in l1_dir.iterdir():
                if not l2_dir.is_dir() or l2_dir.name.startswith("_"):
                    continue
                label = _norm(l2_dir.name)
                aliases = sorted(_label_aliases(label), key=lambda x: (-len(x.split()), x))
                if aliases:
                    records.append({"l1": l1_dir.name, "l2": l2_dir.name, "aliases": tuple(aliases)})

    brands = []
    if BRANDS_PATH.exists():
        try:
            raw = json.loads(BRANDS_PATH.read_text())
        except Exception:
            raw = []
        for brand in raw if isinstance(raw, list) else []:
            clean = _norm(brand)
            if 2 <= len(clean) <= 30 and not clean.startswith("#"):
                brands.append(clean)

    brands = sorted(set(brands), key=lambda x: (-len(x), x))
    return tuple(records), tuple(brands)


def _contains_phrase(haystack: str, phrase: str) -> bool:
    return f" {phrase} " in f" {haystack} "


def _safe_category_aliases(label: str) -> list[str]:
    """Clean a routed category label into product-like retrieval aliases."""
    label_norm = _norm(label)
    if not label_norm or label_norm in BAD_CATEGORY_QUERY_LABELS:
        return []
    aliases = sorted(_label_aliases(label_norm), key=lambda x: (-len(x.split()), x))
    out = []
    seen = set()
    for alias in aliases:
        clean = _clean_alias(alias)
        if not clean or clean in seen:
            continue
        tokens = clean.split()
        if len(tokens) == 1 and tokens[0] not in HEAD_SAFE_TOKENS:
            continue
        seen.add(clean)
        out.append(clean)
    return out


def category_query_variants(
    category_guess: dict | None,
    *,
    category_locked: bool = False,
    min_confidence: float = 0.95,
    limit: int = 4,
) -> list[str]:
    """
    Return product-like aliases from a high-confidence routed category.

    This is recall-only query text. It does not change the route, lock scope,
    or filter candidates. Broad L1 labels are intentionally ignored unless the
    route is locked and the label cleans to a safe product head.
    """
    category_guess = category_guess or {}
    try:
        confidence = float(category_guess.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    if confidence < min_confidence:
        return []

    labels = [category_guess.get("l2")]

    variants = []
    for label in labels:
        variants.extend(_safe_category_aliases(label))

    out = []
    seen = set()
    for variant in variants:
        if variant in seen:
            continue
        seen.add(variant)
        out.append(variant)
        if len(out) >= limit:
            break
    return out


def explicit_product_aliases(text: str, *, limit: int = 6) -> list[str]:
    """
    Return catalog product aliases explicitly present in text.

    NER uses this as a deterministic backup to the hand-curated product phrase
    list. It only fires when the alias words are actually present in the user
    text, so it is not a category inference or label-based shortcut.
    """
    haystack = _norm(text)
    if not haystack:
        return []
    records, _brands = _catalog_records()
    hay_tokens = set(_tokens(haystack))
    matches = []
    for record in records:
        for alias in record["aliases"]:
            alias_tokens = set(alias.split())
            exact = _contains_phrase(haystack, alias)
            subset = alias_tokens and alias_tokens <= hay_tokens and len(alias_tokens) >= 2
            if exact or subset:
                matches.append((haystack.find(alias) if exact else 10_000, -len(alias_tokens), alias))
                break
    matches.sort()

    out = []
    seen = set()
    for _pos, _neg_len, alias in matches:
        clean = _clean_alias(alias)
        if not clean or clean in seen:
            continue
        seen.add(clean)
        out.append(clean)
        if len(out) >= limit:
            break
    return out


def catalog_query_variants(
    text: str,
    *,
    product_type: str = "",
    category_guess: dict | None = None,
    category_locked: bool = False,
    limit: int = 5,
) -> list[str]:
    """
    Return short catalog-derived recall variants for the current query text.

    Variants are conservative:
    - exact phrase/alias hits from index labels are allowed;
    - brand-only variants are avoided unless paired with product/category text;
    - category labels are not used as query text; earlier eval traces showed
      even locked taxonomy labels can drift retrieval when the route is wrong.
    """
    haystack = _norm(" ".join([text or "", product_type or ""]))
    if not haystack:
        return []
    records, brands = _catalog_records()
    variants = []
    hay_tokens = set(_tokens(haystack))

    matched_brands = [b for b in brands[:3000] if _contains_phrase(haystack, b)][:2]

    for record in records:
        for alias in record["aliases"]:
            alias_tokens = set(alias.split())
            exact = _contains_phrase(haystack, alias)
            subset = alias_tokens and alias_tokens <= hay_tokens and len(alias_tokens) >= 2
            head = len(alias_tokens) == 1 and next(iter(alias_tokens)) in hay_tokens and next(iter(alias_tokens)) in HEAD_SAFE_TOKENS
            if exact or subset or head:
                variants.append(alias)
                if matched_brands:
                    variants.append(f"{matched_brands[0]} {alias}")
                break

    if product_type and matched_brands:
        variants.insert(0, f"{matched_brands[0]} {_clean_alias(product_type) or product_type}")

    variants.extend(category_query_variants(
        category_guess,
        category_locked=category_locked,
        min_confidence=0.95,
        limit=3,
    ))

    out = []
    seen = set()
    for variant in variants:
        clean = _clean_alias(variant)
        if not clean or clean in seen:
            continue
        seen.add(clean)
        out.append(clean)
        if len(out) >= limit:
            break
    return out
