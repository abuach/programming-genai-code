"""Regenerate chapters/images/mm/flowchart.png (Figure: API request handling flowchart).

The original render stacked every node in one vertical column, so each decision
diamond's "Yes"/"No" branches collided with the box directly below it: the branch
arrows cut diagonally into the boxes and the "Yes"/"No" labels overlapped the box
text. This lays the three branch outcomes (401, 429, cached response) out to the
right, so the main path runs straight down the spine and nothing overlaps.

The logic and labels must stay as-is: the frozen ask_image cell in mm.ipynb traces
this exact flowchart (invalid token -> 401, exceeded rate limit -> 429, cache hit
returns immediately, cache miss -> database -> cache -> 200 OK) and the prose after
it describes that trace. Run: .venv/bin/python scripts/gen_flowchart_png.py

Diamonds are sized so their two-line labels fit inside the shape: a diamond
narrows away from its vertical center, so each text line (offset ~0.17 units
from center) only gets ~77% of the full width. At 2.3 wide the labels clipped
through the edges; 3.4 leaves a comfortable margin without shrinking the font.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon

OUT = "chapters/images/mm/flowchart.png"

# semantic palette: start / decision / error / success / neutral
BLUE = dict(fill="#cfe2ff", edge="#2563eb", text="#1e40af")
AMBER = dict(fill="#fde7ad", edge="#b45309", text="#92400e")
ERROR = dict(fill="#fcdcda", edge="#b91c1c", text="#991b1b")
GREEN = dict(fill="#c9ecdd", edge="#047857", text="#065f46")
GRAY = dict(fill="#eceef1", edge="#475569", text="#334155")
ARROW = "#3f4a5c"

BW, BH = 3.3, 1.0           # box width / height
DW, DH = 3.4, 1.5           # diamond width / height
XS, XB = 0.0, 5.0           # spine column x / branch column x


def box(ax, cx, cy, label, style, w=BW, h=BH, fs=14):
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0,rounding_size=0.14",
        linewidth=2, facecolor=style["fill"], edgecolor=style["edge"]))
    ax.text(cx, cy, label, ha="center", va="center",
            color=style["text"], fontsize=fs, fontweight="bold")
    return dict(top=(cx, cy + h / 2), bottom=(cx, cy - h / 2),
                left=(cx - w / 2, cy), right=(cx + w / 2, cy))


def diamond(ax, cx, cy, label, style, w=DW, h=DH, fs=14):
    pts = [(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2), (cx - w / 2, cy)]
    ax.add_patch(Polygon(pts, closed=True, linewidth=2,
                         facecolor=style["fill"], edgecolor=style["edge"]))
    ax.text(cx, cy, label, ha="center", va="center",
            color=style["text"], fontsize=fs, fontweight="bold")
    return dict(top=(cx, cy + h / 2), bottom=(cx, cy - h / 2),
                left=(cx - w / 2, cy), right=(cx + w / 2, cy))


def arrow(ax, p0, p1):
    ax.annotate("", xy=p1, xytext=p0, arrowprops=dict(
        arrowstyle="-|>", color=ARROW, linewidth=2, shrinkA=0, shrinkB=0,
        mutation_scale=20))


def tag(ax, x, y, text):
    ax.text(x, y, text, ha="center", va="center", fontsize=12.5,
            color="#334155", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none"))


fig, ax = plt.subplots(figsize=(7.2, 11.2))

# main spine, top to bottom
start = box(ax, XS, 13.0, "API Request\nReceived", BLUE)
jwt = diamond(ax, XS, 11.0, "Valid JWT\nToken?", AMBER)
rate = diamond(ax, XS, 8.6, "Rate Limit\nExceeded?", AMBER)
cache = diamond(ax, XS, 6.2, "Cache\nHit?", AMBER)
query = box(ax, XS, 4.0, "Query Database", GRAY)
store = box(ax, XS, 2.2, "Store in Cache", AMBER)
ok = box(ax, XS, 0.4, "200 OK\nReturn Response", GREEN)

# side outcomes, aligned with the diamond that branches to them
unauth = box(ax, XB, 11.0, "401\nUnauthorized", ERROR, w=2.9)
toomany = box(ax, XB, 8.6, "429 Too Many\nRequests", ERROR, w=2.9)
cached = box(ax, XB, 6.2, "Return Cached\nResponse", GREEN, w=2.9)

# spine arrows (the path that continues downward)
arrow(ax, start["bottom"], jwt["top"])
arrow(ax, jwt["bottom"], rate["top"])
arrow(ax, rate["bottom"], cache["top"])
arrow(ax, cache["bottom"], query["top"])
arrow(ax, query["bottom"], store["top"])
arrow(ax, store["bottom"], ok["top"])

# branch arrows (the outcomes off to the side)
arrow(ax, jwt["right"], unauth["left"])
arrow(ax, rate["right"], toomany["left"])
arrow(ax, cache["right"], cached["left"])

# edge labels, each in its own clear gap
xmid = (DW / 2 + (XB - 2.9 / 2)) / 2      # midpoint of every horizontal branch arrow
tag(ax, XS + 0.34, 10.0, "Yes")           # token valid -> continue down the spine
tag(ax, XS + 0.34, 7.6, "No")             # under the limit -> continue
tag(ax, XS + 0.34, 5.2, "No")             # cache miss -> continue
tag(ax, xmid, 11.36, "No")                # token invalid -> 401
tag(ax, xmid, 8.96, "Yes")                # over the limit -> 429
tag(ax, xmid, 6.56, "Yes")                # cache hit -> cached response

ax.set_title("API Request Handling: Decision Flowchart",
             fontsize=18, fontweight="bold", color="#1f2937", pad=16)
ax.set_xlim(-1.8, 6.6)
ax.set_ylim(-0.6, 14.0)
ax.set_aspect("equal")
ax.axis("off")

fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
print(f"wrote {OUT}")
