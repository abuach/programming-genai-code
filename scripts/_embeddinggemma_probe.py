"""Probe: which embedding models place cross-lingual meaning together?

The Semantics chapter's thesis is that meaning is a location in vector space. A
multilingual embedding model extends it: a sentence and its translation should
land in nearly the same place, even across scripts. We score each (English,
translation) pair under three models, with an unrelated cross-lingual control and
a same-language paraphrase as anchors:

  - nomic-embed-text        : the English-only incumbent (v1.5)
  - nomic-embed-text-v2-moe : Nomic's multilingual MoE (100+ languages)
  - embeddinggemma:300m     : Google's multilingual embedder

nomic v2 is built around task prefixes (search_query:/search_document:), while the
book's genai.embed calls raw. We test v2 both ways so a low raw score isn't mistaken
for a capability gap (cf. the codeqwen camelCase artifact).

Run: uv run python scripts/_embeddinggemma_probe.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "chapters"))
from genai.embed import embed, similarity

EN = "The cat sat on the mat."
TRANSLATIONS = [
    ("French",  "Le chat était assis sur le tapis."),
    ("Spanish", "El gato se sentó en la alfombra."),
    ("German",  "Die Katze saß auf der Matte."),
    ("Chinese", "猫坐在垫子上。"),
    ("Swahili", "Paka aliketi juu ya mkeka."),
]
PARAPHRASE = ("English paraphrase", "A feline rested on the rug.")
UNRELATED  = ("Spanish unrelated",  "La bolsa de valores cayó hoy.")  # the stock market fell today

# (label, model, prefix) — prefix is prepended to both sides before embedding.
RUNS = [
    ("nomic-embed-text (English, raw)",      "nomic-embed-text",        ""),
    ("nomic-embed-text-v2-moe (raw)",        "nomic-embed-text-v2-moe", ""),
    ("nomic-embed-text-v2-moe (search_query)","nomic-embed-text-v2-moe", "search_query: "),
    ("embeddinggemma:300m (raw)",            "embeddinggemma:300m",     ""),
]


def score(model, prefix, a, b):
    return similarity(embed(prefix + a, model), embed(prefix + b, model))


for label, model, prefix in RUNS:
    print(f"\n=== {label} : cosine of '{EN}' vs ... ===")
    print(f"  {PARAPHRASE[0]:22} {score(model, prefix, EN, PARAPHRASE[1]):+.3f}   (same-language anchor)")
    for lang, txt in TRANSLATIONS:
        print(f"  {lang+' translation':22} {score(model, prefix, EN, txt):+.3f}")
    print(f"  {UNRELATED[0]:22} {score(model, prefix, EN, UNRELATED[1]):+.3f}   (control: should stay low)")
