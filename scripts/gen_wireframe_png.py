"""Regenerate chapters/images/mm/wireframe.png (Figure: mobile app UI wireframe).

The original render was 900x1460 px with small fonts, so at the figure's
on-page width the labels were unreadable. It also drew emoji the font lacks
(header icon, search icon, three nav icons all rendered as tofu boxes),
double-printed the "Tasks" nav label, and left the middle 40% of the phone
empty. This render uses bigger fonts relative to image width, draws every
icon from matplotlib primitives, and trims the dead space.

The content must stay as-is: the frozen ask_image cell in mm.ipynb describes
this exact screen (search, All/Today/In Progress/Done filters, tasks marked
"Due today" / "In progress" / "Tomorrow" / "Blocked", bottom nav with stats
and profile), and the frozen multi-image cell reasons about it against
spec_diagram.png. Run: .venv/bin/python scripts/gen_wireframe_png.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, Arc

OUT = "chapters/images/mm/wireframe.png"

BLUE, BLUE_DARK = "#2563eb", "#1d4ed8"
SLATE, SLATE_LIGHT, INK = "#475569", "#cbd5e1", "#1f2937"
CARDS = [  # task label, status pill, card fill, card edge, pill text color
    ("Design system review", "Due today",   "#dbeafe", "#93c5fd", "#1d4ed8"),
    ("API integration tests", "In progress", "#d1fae5", "#6ee7b7", "#047857"),
    ("Update documentation", "Tomorrow",    "#fef3c7", "#fcd34d", "#b45309"),
    ("Deploy to staging",    "Blocked",     "#fee2e2", "#fca5a5", "#b91c1c"),
]


def rbox(ax, x, y, w, h, fill, edge, lw=1.4, r=0.16):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
        linewidth=lw, facecolor=fill, edgecolor=edge))


fig, ax = plt.subplots(figsize=(4.3, 5.9))

# phone body and screen edges
rbox(ax, 0.2, 0.2, 9.6, 12.8, "white", "#64748b", lw=2.5, r=0.4)
L, R = 0.55, 9.45          # inner screen x bounds
W = R - L

# status bar
rbox(ax, L, 12.1, W, 0.62, "#e5e7eb", "none")
ax.text(L + 0.35, 12.41, "9:41 AM", fontsize=9.5, color=SLATE, va="center")
for i in range(3):
    ax.add_patch(Circle((R - 0.95 + i * 0.28, 12.41), 0.06, color=SLATE))

# header bar: drawn check-square logo + app name
rbox(ax, L, 10.85, W, 1.05, BLUE, BLUE_DARK)
rbox(ax, 3.05, 11.16, 0.44, 0.44, "none", "white", lw=1.8, r=0.08)
ax.plot([3.15, 3.25, 3.44], [11.38, 11.26, 11.52], color="white", lw=2.0)
ax.text(3.75, 11.38, "Task Manager", fontsize=13.5, fontweight="bold",
        color="white", va="center")

# search field: drawn magnifier + placeholder
rbox(ax, L + 0.15, 9.65, W - 0.3, 0.92, "white", SLATE_LIGHT)
ax.add_patch(Circle((1.25, 10.16), 0.13, fill=False, color="#94a3b8", lw=1.6))
ax.plot([1.34, 1.48], [10.05, 9.92], color="#94a3b8", lw=1.6)
ax.text(1.75, 10.11, "Search tasks...", fontsize=11, color="#94a3b8",
        va="center")

# filter chips, "All" active, widths proportional to label length
x = L + 0.15
for i, (label, chip_w) in enumerate(zip(
        ["All", "Today", "In Progress", "Done"], [1.35, 2.0, 2.9, 1.7])):
    active = i == 0
    rbox(ax, x, 8.55, chip_w, 0.78,
         BLUE if active else "#f1f5f9", BLUE_DARK if active else SLATE_LIGHT)
    ax.text(x + chip_w / 2, 8.94, label, fontsize=10,
            fontweight="bold" if active else "normal",
            color="white" if active else "#334155", ha="center", va="center")
    x += chip_w + 0.2

# task cards with status pills
for i, (label, status, fill, edge, txt) in enumerate(CARDS):
    y = 7.0 - i * 1.36
    rbox(ax, L + 0.15, y, W - 0.3, 1.12, fill, edge)
    ax.text(L + 0.5, y + 0.56, label, fontsize=11, color=INK, va="center")
    rbox(ax, R - 2.8, y + 0.21, 2.35, 0.7, "white", edge)
    ax.text(R - 1.625, y + 0.56, status, fontsize=9, color=txt,
            ha="center", va="center")

# bottom nav: drawn icons (house / check / bars / person), "Tasks" active
rbox(ax, L, 0.5, W, 1.35, "#f1f5f9", "none")
ax.plot([L, R], [1.85, 1.85], color="#e2e8f0", lw=1.2)
for cx, label in zip([1.66, 3.89, 6.11, 8.34],
                     ["Home", "Tasks", "Stats", "Profile"]):
    active = label == "Tasks"
    c = BLUE if active else SLATE
    if label == "Home":
        ax.plot([cx - 0.22, cx - 0.22, cx, cx + 0.22, cx + 0.22, cx - 0.22],
                [1.08, 1.36, 1.54, 1.36, 1.08, 1.08], color=c, lw=1.8)
    elif label == "Tasks":
        ax.plot([cx - 0.2, cx - 0.04, cx + 0.24],
                [1.3, 1.1, 1.5], color=c, lw=2.6)
    elif label == "Stats":
        for j, h in enumerate([0.18, 0.3, 0.42]):
            ax.add_patch(Rectangle((cx - 0.24 + j * 0.18, 1.08), 0.12, h,
                                   color=c))
    else:
        ax.add_patch(Circle((cx, 1.4), 0.1, fill=False, color=c, lw=1.8))
        ax.add_patch(Arc((cx, 1.02), 0.46, 0.42, theta1=20, theta2=160,
                         color=c, lw=1.8))
    ax.text(cx, 0.78, label, fontsize=9.5, color=c, ha="center", va="center",
            fontweight="bold" if active else "normal")

ax.set_title("Task Manager App: UI Wireframe",
             fontsize=13, fontweight="bold", color=INK, pad=12)
ax.set_xlim(-0.1, 10.1)
ax.set_ylim(0.0, 13.2)
ax.set_aspect("equal")
ax.axis("off")

fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
print(f"wrote {OUT}")
