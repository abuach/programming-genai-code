"""Probe: which original sentence + model gives a clean 'it -> antecedent'
attention shift when we swap a single word? Data decides the demo.

Tests a few original Winograd-style pairs on a bidirectional model (BERT) and
a decoder (gpt2). For each sentence we report, for the pronoun token 'it', how
much attention lands on each earlier word (mean over all heads in a layer, and
mean over all layers/heads). We want: in sentence A 'it' points at noun_1, and
after a one-word swap in sentence B it points at noun_2.
"""
import torch
import numpy as np


PAIRS = [
    # (label, sentence_A, sentence_B, noun_1, noun_2)
    ("cat/laptop",
     "The cat sat on the laptop because it was tired.",
     "The cat sat on the laptop because it was warm.",
     "cat", "laptop"),
    ("trophy/suitcase-orig (kite/drone)",
     "The kite outflew the drone because it was light.",
     "The kite outflew the drone because it was heavy.",
     "kite", "drone"),
    ("novel/textbook",
     "The novel outsold the textbook because it was gripping.",
     "The novel outsold the textbook because it was dull.",
     "novel", "textbook"),
    ("river/canyon",
     "The river carved the canyon because it was relentless.",
     "The river carved the canyon because it was soft.",
     "river", "canyon"),
    ("compiler/script",
     "The compiler rejected the script because it was strict.",
     "The compiler rejected the script because it was sloppy.",
     "compiler", "script"),
]


def bert_attn_to_it(sentence, nouns):
    from transformers import BertTokenizer, BertModel
    tok = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased", output_attentions=True).eval()
    enc = tok(sentence, return_tensors="pt")
    with torch.no_grad():
        out = model(**enc)
    toks = tok.convert_ids_to_tokens(enc["input_ids"][0])
    it_idx = [i for i, t in enumerate(toks) if t == "it"]
    if not it_idx:
        return toks, None, {}
    it = it_idx[0]
    # attentions: tuple(layers) of [batch, heads, q, k]
    A = torch.stack(out.attentions)[:, 0]            # [layers, heads, q, k]
    # report per-layer (mean over heads) and overall (mean over all)
    results = {}
    overall = A.mean(dim=(0, 1))[it].numpy()         # [k]
    results["overall"] = {n: float(overall[toks.index(n)]) for n in nouns if n in toks}
    for L in range(A.shape[0]):
        row = A[L].mean(dim=0)[it].numpy()
        results[f"L{L}"] = {n: float(row[toks.index(n)]) for n in nouns if n in toks}
    return toks, it, results


def main():
    print("=" * 70)
    print("BERT (bidirectional): attention FROM 'it' TO each noun")
    print("=" * 70)
    for label, sa, sb, n1, n2 in PAIRS:
        print(f"\n### {label}")
        for tag, s in [("A", sa), ("B", sb)]:
            toks, it, res = bert_attn_to_it(s, [n1, n2])
            if it is None:
                print(f"  [{tag}] NO 'it' token in: {toks}")
                continue
            ov = res["overall"]
            # find the layer with the cleanest separation toward the expected noun
            expected = n1 if tag == "A" else n2
            best_layer, best_gap = None, -9
            for k, v in res.items():
                if k == "overall" or n1 not in v or n2 not in v:
                    continue
                gap = v[expected] - v[(n2 if expected == n1 else n1)]
                if gap > best_gap:
                    best_layer, best_gap = k, gap
            print(f"  [{tag}] expect it->{expected:9s} | overall "
                  f"{n1}={ov.get(n1,0):.3f} {n2}={ov.get(n2,0):.3f} "
                  f"| best {best_layer} {res[best_layer]}")


if __name__ == "__main__":
    main()


def dump_full(sentence):
    from transformers import BertTokenizer, BertModel
    tok = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased", output_attentions=True).eval()
    enc = tok(sentence, return_tensors="pt")
    with torch.no_grad():
        out = model(**enc)
    toks = tok.convert_ids_to_tokens(enc["input_ids"][0])
    it = toks.index("it")
    A = torch.stack(out.attentions)[:, 0]
    overall = A.mean(dim=(0, 1))[it].numpy()
    L10 = A[10].mean(dim=0)[it].numpy()
    print("\nFULL attention from 'it' :", sentence)
    for i, t in enumerate(toks):
        print(f"   {t:12s} overall={overall[i]:.3f}  L10={L10[i]:.3f}")
