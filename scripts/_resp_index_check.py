"""Index-term coverage checker for a chapter: which patterns from
scripts/index_terms.txt match the chapter's markdown prose (code fences,
inline code, and {cite} roles stripped, approximating build_pdf's
prose-only injection). Used to verify a prose trim drops no index entry.

Usage: python scripts/_resp_index_check.py chapters/responsible.ipynb [slug]
"""
import json
import re
import sys
from pathlib import Path

nb_path = Path(sys.argv[1])
slug = sys.argv[2] if len(sys.argv) > 2 else nb_path.stem

terms = []
for line in Path("scripts/index_terms.txt").read_text().splitlines():
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    parts = [p.strip() for p in line.split("::")]
    pattern = parts[0]
    key = parts[1] if len(parts) > 1 and parts[1] and not parts[1].startswith(("slugs=", "cs")) else pattern
    slugs = None
    cs = False
    for p in parts[1:]:
        if p.startswith("slugs="):
            slugs = [s.strip() for s in p[len("slugs="):].split(",")]
        elif p == "cs":
            cs = True
    if slugs is not None and slug not in slugs:
        continue
    terms.append((pattern, key, cs))

def prose_lines(src):
    """Keep prose, including text inside {admonition}-style directive fences;
    drop lines inside plain code fences and the fence/option lines themselves."""
    out, stack = [], []
    for line in src.splitlines():
        m = re.match(r"\s*(`{3,})(.*)$", line)
        if m:
            ticks, info = len(m.group(1)), m.group(2).strip()
            if stack and ticks >= stack[-1][0]:
                stack.pop()
            else:
                stack.append((ticks, info.startswith("{")))
            continue
        if line.strip().startswith(":class:"):
            continue
        if all(is_directive for _, is_directive in stack):
            out.append(line)
    return "\n".join(out)


nb = json.loads(nb_path.read_text(encoding="utf-8"))
chunks = []
for c in nb["cells"]:
    if c["cell_type"] != "markdown":
        continue
    src = prose_lines("".join(c["source"]))
    src = re.sub(r"\{cite\}`[^`]*`", " ", src)          # citation roles
    src = re.sub(r"`[^`]*`", " ", src)                  # inline code
    chunks.append(src)
text = "\n".join(chunks)

matched = []
for pattern, key, cs in terms:
    flags = 0 if cs else re.IGNORECASE
    if re.search(rf"\b(?:{pattern})\b", text, flags):
        matched.append(key)
for key in sorted(set(matched)):
    print(key)
print(f"-- {len(set(matched))} index keys matched in {slug}", file=sys.stderr)
