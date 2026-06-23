"""Reproducer for the Control Panel sampling figure (genai.viz.plot_sampling_controls).

Documents why the figure uses the prompt, model, and (k, p, n) it does. We want a
real next-token distribution that is (a) spread enough for temperature reshaping
to show on the left panel, and (b) where top-k and top-p draw their lines in
different places on the right panel. "She opened the door and saw a" on
llama3.2:latest (the same prompt + model the Introduction uses) fits: the
top guesses are clean content words and the model is genuinely unsure what she
saw, so with k=3 / p=0.9 over the top n=5, top-k keeps 3 while top-p widens to 4.
Pulled at temperature 0, so the snapshot is reproducible.
"""
import sys
sys.path.insert(0, "chapters")
from genai import next_token_distribution

PROMPT, MODEL, K, P, N = "She opened the door and saw a", "llama3.2:latest", 3, 0.9, 5

dist = next_token_distribution(PROMPT, model=MODEL, top_k=10)[:N]
base = [pr for _, pr in dist]
base = [pr / sum(base) for pr in base]          # renormalize over the shown candidates
cum, top_p_n = 0.0, 0
for i, pr in enumerate(base):
    cum += pr
    if cum >= P and not top_p_n:
        top_p_n = i + 1

print(f"{PROMPT!r}  ({MODEL})")
print(f"top-k={K} keeps {K}   top-p={P} keeps {top_p_n}   (n={N} shown)\n")
cum = 0.0
for i, ((tok, _), pr) in enumerate(zip(dist, base)):
    cum += pr
    marks = " ".join(m for m, on in (("k", i < K), ("p", i < top_p_n)) if on)
    print(f"  {i+1}. {tok!r:12} {pr*100:5.1f}%  cum={cum*100:5.1f}%  {marks}")
