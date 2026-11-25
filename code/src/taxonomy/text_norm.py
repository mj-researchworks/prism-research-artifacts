"""
Shared tokenizer for taxonomy routing — used by both the offline route-index
builder (scripts/10_build_route_index.py) and CategoryRouter at query time.
Must stay identical between the two, or the vocab built offline won't match
tokens produced from a live query. Import this rather than reimplementing.
"""

import re

STOPWORDS = {
    "the", "a", "an", "and", "or", "for", "of", "in", "on", "with", "to",
    "is", "are", "this", "that", "it", "its", "your", "you", "i", "as",
    "by", "be", "was", "were", "at", "from", "not", "no", "up", "so",
}

# Irregular plurals common enough in the catalog domain to matter for routing
# (e.g. "mice" in the L2 label "PC Gaming Mice" must match a user's "mouse").
IRREGULAR_PLURALS = {
    "mice": "mouse",
    "feet": "foot",
    "teeth": "tooth",
    "children": "child",
    "geese": "goose",
    "knives": "knife",
    "wives": "wife",
    "lives": "life",
    "leaves": "leaf",
    "shelves": "shelf",
    "loaves": "loaf",
    "scarves": "scarf",
}


def _stem(word: str) -> str:
    if word in IRREGULAR_PLURALS:
        return IRREGULAR_PLURALS[word]
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith(("xes", "ses", "ches", "shes")):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def tokenize(text: str, bigrams: bool = True) -> list:
    """
    Unigrams + adjacent-pair bigrams (joined with "_"), both stemmed.
    Bigrams matter for routing: "contact lens" and "camera lens" both reduce
    to the unigram "lens" alone, which is exactly what caused "contact
    lenses" queries to wrongly match Mirrorless Cameras. The bigram
    "contact_lens" doesn't collide with "camera_lens", so keeping phrase
    context resolves that class of mismatch.
    """
    text = text.lower()
    words = re.findall(r"[a-z0-9]+", text)
    words = [w for w in words if w not in STOPWORDS and len(w) > 1]
    stems = [_stem(w) for w in words]
    if not bigrams:
        return stems
    bigram_terms = [f"{a}_{b}" for a, b in zip(stems, stems[1:])]
    return stems + bigram_terms
