"""THROWAWAY probe: find an honest substrate for a Chain-of-Verification (CoVe) demo.

CoVe {cite}`dhuliawala2023cove`: the model writes a baseline DRAFT answer (which
for a list question often over-generates and slips in a wrong entry), then PLANS a
focused yes/no verification question for each listed item, EXECUTES each one
INDEPENDENTLY (factored: no draft in context, so it cannot just re-read its own
mistake), and REVISES by keeping only the items its own factored check confirmed.
It should BEAT the draft when the draft over-generates but the atomized "Does X
relate to Y?" question is something the model reliably gets right on its own, and
it CANNOT help when the model is stably wrong even one-on-one (the factored check
repeats the mistake) -- the exact mirror of Reflexion / Self-Consistency.

This probe measures, per (model, question):
  - draft P  : list precision of the baseline answer (correct items / items listed)
  - cove  P  : list precision after dropping items the factored check rejected
  - cove  R  : recall after the drop (correct kept / total gold) -- the COST, since
               CoVe sometimes drops a correct item it cannot confirm

At M=1 (triage, temp 0) it prints the actual draft list and each kept/dropped
decision so we can eyeball where the honest gap lives before baking anything.

Run: .venv/bin/python scripts/_cove_probe.py [M] [model ...]
"""
import sys, re, time
sys.path.insert(0, "chapters")
import ollama
from genai.llm import SERVER

_client = ollama.Client(host=SERVER)

# Curated list questions over STABLE, verifiable facts (geography, well-known sets)
# so the gold set is uncontroversial and a factored "Does X border Y?" check is a
# crisp yes/no the model knows in isolation. Each: (label, draft_prompt,
# verify_template, gold). Gold carries every accepted surface form (czechia AND
# czech republic) so the canonical match is forgiving. The high-neighbour sets
# (Brazil, China, Germany) are the over-generation candidates; the short ones
# (Portugal, Switzerland, Lakes) are ceiling candidates.
QUESTIONS = [
    ("Brazil",
     "Name the South American countries that share a land border with Brazil.",
     "Does Brazil share a land border with {item}?",
     {"argentina", "bolivia", "colombia", "guyana", "paraguay", "peru",
      "suriname", "uruguay", "venezuela", "french guiana"}),
    ("France",
     "Name the European countries that share a land border with France.",
     "Does France share a land border with {item}?",
     {"spain", "andorra", "monaco", "italy", "switzerland", "germany",
      "luxembourg", "belgium"}),
    ("Germany",
     "Name the countries that share a land border with Germany.",
     "Does Germany share a land border with {item}?",
     {"denmark", "poland", "czech republic", "czechia", "austria", "switzerland",
      "france", "luxembourg", "belgium", "netherlands"}),
    ("China",
     "Name the countries that share a land border with China.",
     "Does China share a land border with {item}?",
     {"afghanistan", "bhutan", "india", "kazakhstan", "north korea", "kyrgyzstan",
      "laos", "mongolia", "myanmar", "burma", "nepal", "pakistan", "russia",
      "tajikistan", "vietnam"}),
    ("Mexico",
     "Name the U.S. states that share a border with Mexico.",
     "Does the U.S. state of {item} share a border with Mexico?",
     {"california", "arizona", "new mexico", "texas"}),
    ("Switzerland",
     "Name the countries that share a land border with Switzerland.",
     "Does Switzerland share a land border with {item}?",
     {"france", "germany", "austria", "italy", "liechtenstein"}),
    ("Lakes",
     "Name the Great Lakes of North America.",
     "Is {item} one of the five Great Lakes of North America?",
     {"superior", "michigan", "huron", "erie", "ontario"}),
    ("Portugal",
     "Name the countries that share a land border with Portugal.",
     "Does Portugal share a land border with {item}?",
     {"spain"}),
]

# The draft asks for a bare comma list (where over-generation shows up cleanly).
# The verify must give the model ROOM to answer: forcing a single yes/no token
# collapses gemma4 to "no" on everything (even "Is Superior a Great Lake?"), so we
# let it answer in a sentence and read the first yes/no out of it.
_DRAFT = " Reply with only the names, separated by commas, and nothing else."
_VERIFY = " Answer yes or no, then give a one sentence reason."


def _canon(name: str) -> str:
    """Lowercase, strip punctuation/articles, collapse spaces so 'the Netherlands'
    and 'Netherlands.' both reduce to 'netherlands'."""
    s = name.lower().strip().strip(".")
    s = re.sub(r"^(the|a)\s+", "", s)
    return re.sub(r"\s+", " ", s).strip()


def _items(reply: str) -> list:
    """Split a comma/newline list reply into canonical item names (deduped, in
    order, blanks dropped)."""
    raw = re.split(r"[,\n]", reply)
    out = []
    for tok in raw:
        c = _canon(re.sub(r"^[\-\*\d\.\)\s]+", "", tok))
        if c and c not in out:
            out.append(c)
    return out


def draft(question: str, model: str, temp: float) -> list:
    """Step 1: the baseline list answer, parsed into canonical items."""
    resp = _client.chat(model=model, think=False,
                        messages=[{"role": "user", "content": question + _DRAFT}],
                        options={"num_predict": 120, "temperature": temp})
    return _items(resp["message"]["content"] or "")


def verify(item: str, template: str, model: str, temp: float) -> bool:
    """Step 2-3: form a focused yes/no question for one item and answer it on its
    OWN (factored: no draft in context). True iff the model says yes."""
    q = template.format(item=item) + _VERIFY
    resp = _client.chat(model=model, think=False,
                        messages=[{"role": "user", "content": q}],
                        options={"num_predict": 45, "temperature": temp})
    m = re.search(r"\b(yes|no)\b", (resp["message"]["content"] or "").lower())
    return m is not None and m.group(1) == "yes"


def _precision(items: list, gold: set) -> float:
    return sum(i in gold for i in items) / len(items) if items else 0.0


def _recall(items: list, gold: set) -> float:
    # gold may hold synonyms (czechia/czech republic); recall counts distinct hits
    # but never above 1.0, so use the smaller "true" size estimate generously.
    hit = sum(i in gold for i in items)
    return min(1.0, hit / max(1, _gold_size(gold)))


# Distinct real entities, collapsing known synonym pairs so recall denominators are
# honest (czechia==czech republic, myanmar==burma).
_SYNONYMS = [{"czech republic", "czechia"}, {"myanmar", "burma"}]


def _gold_size(gold: set) -> int:
    n = len(gold)
    for pair in _SYNONYMS:
        if pair <= gold:
            n -= 1
    return n


def cove(question, template, gold, model, temp):
    """Run the four factored steps; return (draft_items, kept_items)."""
    d = draft(question, model, temp)
    kept = [it for it in d if verify(it, template, model, temp)]
    return d, kept


def triage(model):
    print(f"\n===== {model}  (triage M=1, temp=0) =====", flush=True)
    for label, q, tmpl, gold in QUESTIONS:
        d, kept = cove(q, tmpl, gold, model, 0.0)
        dropped = [it for it in d if it not in kept]
        print(f"\n## {label}  (gold has {_gold_size(gold)})")
        print(f"  draft   P={_precision(d, gold):4.0%}  {d}")
        print(f"  kept    P={_precision(kept, gold):4.0%} R={_recall(kept, gold):4.0%}  {kept}")
        print(f"  dropped {dropped}")
        bad_drop = [it for it in dropped if it in gold]
        bad_keep = [it for it in kept if it not in gold]
        if bad_drop:
            print(f"  !! dropped CORRECT items (recall cost): {bad_drop}")
        if bad_keep:
            print(f"  !! kept WRONG items (stable wrong prior): {bad_keep}")


# The subset baked into genai.cove.COVE_STUDY, chosen from triage to span the
# three honest regimes with gemma4: Portugal is the WIN (the draft slips in Morocco
# across the strait, the factored check rejects it, precision climbs), Mexico is the
# CEILING (draft already clean, CoVe a no-op), Brazil is the LIMIT (the model holds
# a stable wrong belief that it does not border Colombia/Bolivia/Peru, so the
# factored check repeats the mistake and drops correct items -- precision flat,
# recall collapses, the boundary where you need real retrieval, not self-checking).
BAKE = ["Portugal", "Mexico", "Brazil"]


def bake(model, M, temp=0.7, only=BAKE):
    print(f"\n===== {model}  (M={M}/question, temp={temp}) =====", flush=True)
    for label, q, tmpl, gold in QUESTIONS:
        if only and label not in only:
            continue
        t0 = time.perf_counter()
        dps, cps, drs, crs = [], [], [], []
        for _ in range(M):
            d, kept = cove(q, tmpl, gold, model, temp)
            dps.append(_precision(d, gold)); drs.append(_recall(d, gold))
            cps.append(_precision(kept, gold)); crs.append(_recall(kept, gold))
        dt = time.perf_counter() - t0
        dp, cp, dr, cr = sum(dps)/M, sum(cps)/M, sum(drs)/M, sum(crs)/M
        print(f"{label:12s} P {dp:4.0%}->{cp:4.0%} (gap {cp-dp:+4.0%})   "
              f"R {dr:4.0%}->{cr:4.0%} (gap {cr-dr:+4.0%})   {dt:5.1f}s", flush=True)


if __name__ == "__main__":
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    models = sys.argv[2:] or ["gemma4:latest"]
    for mdl in models:
        if M == 1:
            triage(mdl)
        else:
            bake(mdl, M)
