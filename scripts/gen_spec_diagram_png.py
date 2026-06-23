"""Regenerate chapters/images/mm/spec_diagram.png (Figure: microservices architecture).

The original render spread the boxes across a wide canvas with small fonts and a
lot of dead margin, so at the figure's 80%-of-text-width print size the sublabels
("React / Next.js", "PostgreSQL", "Auth + Rate Limit"...) were illegible. This
redraws the same diagram on a tight canvas with print-sized fonts: on-page font
size = point size x (on-page width / figsize width), so the fix is a narrower
figure and bigger points, not a higher DPI.

The boxes, labels, and edges must stay as-is: the frozen ask_image cell in
mm.ipynb describes this exact diagram (Web/Mobile clients -> API Gateway ->
User/Task/Notif services -> PostgreSQL DBs + Redis cache, Message Queue
decoupling, metrics -> Prometheus/Grafana), and a later frozen cell cross-checks
it against the UI wireframe. Run: .venv/bin/python scripts/gen_spec_diagram_png.py

Edge-label gaps are sized to their labels: a boxed tag needs its gap to fit
the tag plus visible arrow stubs on both sides, or the white label box
swallows the arrow and crowds the node edges. The client gap (1.5) exists so
the HTTPS tags fit on their arrows. The bottom row is decoupled from the
column grid for the same reason: the queue slides left of C3 and the
observability box right-aligns with the DB column so the "metrics" tag fits
between them, and both boxes are wider than BW because their labels
("Message Queue", "Prometheus + Grafana") overflow the standard width.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = "chapters/images/mm/spec_diagram.png"

# semantic palette: clients / gateway / services / data stores / cache / queue
BLUE = dict(fill="#cfe2ff", edge="#2563eb", text="#1e40af")
AMBER = dict(fill="#fde7ad", edge="#b45309", text="#92400e")
GREEN = dict(fill="#c9ecdd", edge="#047857", text="#065f46")
GRAY = dict(fill="#eceef1", edge="#475569", text="#334155")
RED = dict(fill="#fcdcda", edge="#b91c1c", text="#991b1b")
ORANGE = dict(fill="#ffe0c2", edge="#c2410c", text="#9a3412")
ARROW = "#3f4a5c"
SUB = "#475569"

BW, BH = 2.95, 1.1          # box width / height
# column centers; the client gap is widest so the "HTTPS" tags fit on
# their arrows with clear stubs each side
C1 = 1.55
C2 = C1 + BW + 1.5
C3 = C2 + BW + 0.75
C4 = C3 + BW + 1.0


def node(ax, cx, cy, name, sub, style, w=BW, h=BH):
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0,rounding_size=0.14",
        linewidth=2, facecolor=style["fill"], edgecolor=style["edge"]))
    ax.text(cx, cy + 0.21, name, ha="center", va="center",
            color=style["text"], fontsize=12.5, fontweight="bold")
    ax.text(cx, cy - 0.27, sub, ha="center", va="center",
            color=SUB, fontsize=9.8)
    return dict(top=(cx, cy + h / 2), bottom=(cx, cy - h / 2),
                left=(cx - w / 2, cy), right=(cx + w / 2, cy))


def arrow(ax, p0, p1, style="-|>"):
    ax.annotate("", xy=p1, xytext=p0, arrowprops=dict(
        arrowstyle=style, color=ARROW, linewidth=1.8, shrinkA=0, shrinkB=0,
        mutation_scale=16))


def tag(ax, x, y, text, boxed=False):
    bbox = dict(boxstyle="round,pad=0.12", fc="white", ec="none") if boxed else None
    ax.text(x, y, text, ha="center", va="center", fontsize=9.5,
            color="#334155", fontweight="bold", bbox=bbox)


fig, ax = plt.subplots(figsize=(10.35, 5.1))

# clients -> gateway -> services -> data stores, left to right
web = node(ax, C1, 5.1, "Web Client", "React / Next.js", BLUE)
mobile = node(ax, C1, 3.5, "Mobile App", "React Native", BLUE)
gateway = node(ax, C2, 4.3, "API Gateway", "Auth + Rate Limit", AMBER)
user_svc = node(ax, C3, 5.9, "User Service", "REST / gRPC", GREEN)
task_svc = node(ax, C3, 4.3, "Task Service", "REST / gRPC", GREEN)
notif = node(ax, C3, 2.7, "Notif. Service", "WebSocket", GREEN)
users_db = node(ax, C4, 5.9, "Users DB", "PostgreSQL", GRAY)
tasks_db = node(ax, C4, 4.3, "Tasks DB", "PostgreSQL", GRAY)
cache = node(ax, C4, 2.7, "Cache", "Redis", RED)
# bottom row, off the column grid: wider boxes for their long labels,
# slid apart so the "metrics" tag fits on the arrow between them
queue = node(ax, C3 - 1.5, 0.9, "Message Queue", "RabbitMQ / Kafka", ORANGE, w=3.5)
obs = node(ax, C4 + BW / 2 - 1.8, 0.9, "Observability", "Prometheus + Grafana", GRAY, w=3.6)

# clients into the gateway
arrow(ax, web["right"], (gateway["left"][0], 4.55))
arrow(ax, mobile["right"], (gateway["left"][0], 4.05))

# gateway fans out to the three services
arrow(ax, (gateway["right"][0], 4.65), user_svc["left"])
arrow(ax, gateway["right"], task_svc["left"])
arrow(ax, (gateway["right"][0], 3.95), notif["left"])

# each service talks to its data store, both ways
arrow(ax, user_svc["right"], users_db["left"], style="<|-|>")
arrow(ax, task_svc["right"], tasks_db["left"], style="<|-|>")
arrow(ax, notif["right"], cache["left"], style="<|-|>")

# async backbone and monitoring
arrow(ax, notif["bottom"], queue["top"])
arrow(ax, queue["right"], obs["left"])

# edge labels: HTTPS and metrics sit on their arrows (white box),
# REST floats in the empty band beside its
tag(ax, (C1 + C2) / 2, 4.825, "HTTPS", boxed=True)
tag(ax, (C1 + C2) / 2, 3.775, "HTTPS", boxed=True)
tag(ax, C3 - BW / 2 - 0.45, 5.82, "REST")
tag(ax, (queue["right"][0] + obs["left"][0]) / 2, 0.9, "metrics", boxed=True)

ax.set_title("Task Manager — Microservices Architecture",
             fontsize=17, fontweight="bold", color="#1f2937", pad=14)
ax.set_xlim(0.0, C4 + BW / 2 + 0.075)
ax.set_ylim(0.0, 6.7)
ax.set_aspect("equal")
ax.axis("off")

fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
print(f"wrote {OUT}")
