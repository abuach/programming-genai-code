"""Visualization helpers for Programming Generative AI."""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mp

# ── Shared palette ────────────────────────────────────────────────────────────
BLUE, GREEN, ORANGE, RED = "#2563EB", "#10B981", "#F97316", "#EF4444"
LGRAY, MGRAY, DGRAY = "#F3F4F6", "#9CA3AF", "#111827"
PURPLE, TEAL, PINK, AMBER = "#8B5CF6", "#14B8A6", "#EC4899", "#F59E0B"
SLATE = "#475569"


# ── Model zoo: one stable color per contestant across every bake-off ──────────
# Keyed by family substring, checked in order so the specific name wins over the
# general one (functiongemma before gemma, qwen3.5 before qwen, ministral before
# mistral). The weak function-tuned tail is gray on purpose; gpt-oss, the heavy
# reasoning newcomer, gets a sober dark slate to set it apart from the family hues.
_MODEL_COLOR_RULES = [
    ("gpt-oss", SLATE),
    ("functiongemma", MGRAY), ("gemma4", GREEN), ("gemma3", AMBER),
    ("qwen3.5", TEAL), ("qwen2.5", TEAL), ("qwen", BLUE),
    ("ministral", PURPLE), ("mistral", ORANGE),
    ("llama3.2", PINK), ("llama", RED),
]


def short_model(name: str) -> str:
    """Drop the ``:latest`` tag for an axis label but keep size tags like ``:1b``."""
    return name[:-7] if name.endswith(":latest") else name


def model_color(name: str) -> str:
    """The stable bake-off color for a model, by family."""
    low = name.lower()
    for key, col in _MODEL_COLOR_RULES:
        if key in low:
            return col
    return DGRAY


# ── Internal helpers ──────────────────────────────────────────────────────────

def _arr(ax, x1, y1, x2, y2, col=MGRAY, pct=0.28, ls="solid"):
    dx, dy = x2 - x1, y2 - y1
    ax.annotate("", xy=(x2 - pct*dx, y2 - pct*dy), xytext=(x1 + pct*dx, y1 + pct*dy),
                arrowprops=dict(arrowstyle="-|>", color=col, lw=1.2,
                                mutation_scale=10, shrinkA=0, shrinkB=0, linestyle=ls))


def _fbox(ax, cx, cy, hw, hh, fc, ec="none", lw=1.0):
    ax.add_patch(mp.FancyBboxPatch((cx - hw, cy - hh), hw*2, hh*2,
                                   boxstyle="round,pad=0.07", facecolor=fc,
                                   edgecolor=ec, linewidth=lw, zorder=3))


def _label(ax, cx, cy, text, tc=DGRAY, fs=10.5, fw="700"):
    ax.text(cx, cy, text, ha="center", va="center", linespacing=1.2,
            fontsize=fs, fontweight=fw, color=tc, zorder=4)


def _grid_ax(ax):
    ax.set_facecolor("white")
    for sp in ax.spines.values():
        sp.set_color(LGRAY)
    ax.tick_params(colors=MGRAY, labelsize=11)
    ax.yaxis.grid(True, color=LGRAY, zorder=0)
    ax.set_axisbelow(True)


def _save(fig, path):
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")


def _place_labels(ax, coords, labels, fontsize=10.5, color=None):
    """Place point labels using adjustText so they don't overlap.

    coords: (N, 2) array of point positions.
    labels: list of N strings.
    """
    from adjustText import adjust_text
    if color is None:
        color = DGRAY
    arr = np.asarray(coords)
    texts = [ax.text(arr[i, 0], arr[i, 1], lbl,
                     fontsize=fontsize, color=color)
             for i, lbl in enumerate(labels)]
    adjust_text(texts, ax=ax,
                expand=(1.15, 1.4),
                arrowprops=dict(arrowstyle='-', color=LGRAY, lw=0.6))
    return texts


# ── Existing helpers ──────────────────────────────────────────────────────────

def _attention_from(sentence, pronoun, layer, head):
    """Read one attention head: how much the ``pronoun`` token looks back at each
    word of ``sentence``. Returns (words, weights) with scaffold tokens dropped."""
    import torch
    from transformers import BertTokenizer, BertModel

    tok = BertTokenizer.from_pretrained("bert-base-uncased", local_files_only=True)
    model = BertModel.from_pretrained("bert-base-uncased", output_attentions=True,
                                      local_files_only=True).eval()
    enc = tok(sentence, return_tensors="pt")
    with torch.no_grad():
        out = model(**enc)
    toks = tok.convert_ids_to_tokens(enc["input_ids"][0])
    row = out.attentions[layer][0, head, toks.index(pronoun)].numpy()
    words = [(t, w) for t, w in zip(toks, row) if t not in ("[CLS]", "[SEP]")]
    return [t for t, _ in words], np.array([w for _, w in words])


def plot_attention(sentences, pronoun: str = "it", layer: int = 9, head: int = 1,
                   path: str = None, figsize=(8.2, 4.4)):
    """Show where a pronoun looks back, for two sentences that differ by one word.

    For each sentence, read one attention head and plot how much the ``pronoun``
    token attends to every earlier word. The word it leans on hardest is the one
    the model ties the pronoun to, and swapping a single word flips it.
    """
    fig, axes = plt.subplots(len(sentences), 1, figsize=figsize, sharex=True)
    axes = np.atleast_1d(axes)
    for ax, sentence in zip(axes, sentences):
        words, weights = _attention_from(sentence, pronoun, layer, head)
        top = int(np.argmax(weights))
        colors = [GREEN if i == top else LGRAY for i in range(len(words))]
        y = range(len(words))
        ax.barh(y, weights, color=colors, zorder=3)
        ax.set_yticks(y); ax.set_yticklabels(words, fontsize=10)
        ax.invert_yaxis()
        ax.set_title(sentence, fontsize=10.5, fontweight="600", loc="left", color=DGRAY)
        ax.text(weights[top], top, f"  {words[top]}", va="center",
                fontsize=10, fontweight="700", color=GREEN)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(length=0, colors=MGRAY)
    axes[-1].set_xlabel(f'attention from "{pronoun}" to each word', fontsize=10.5)
    plt.tight_layout()
    _save(fig, path)
    plt.show()


def plot_embeddings_2d(words: list, vecs,
                       figsize=(6, 6), title: str = "Word Embeddings (PCA)"):
    """Project high-dimensional embeddings to 2D with PCA and scatter-plot them."""
    from sklearn.decomposition import PCA

    coords = PCA(n_components=2).fit_transform(vecs)
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(coords[:, 0], coords[:, 1], s=80)
    for word, (x, y) in zip(words, coords):
        ax.annotate(word, (x, y), fontsize=14, textcoords="offset points", xytext=(4, 4))
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


# ── Agentic chapter ───────────────────────────────────────────────────────────

def plot_agentic_loop(save_path="images/agentic/agentic_loop.png"):
    """Architecture diagram of the sense-plan-act agentic loop."""
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.set_xlim(0, 10); ax.set_ylim(0.8, 5.2); ax.axis("off")

    nodes = {
        "goal":  (1.4, 3.0, "Goal",            BLUE,  "white"),
        "plan":  (3.6, 4.4, "Plan\n(Planner)", LGRAY, DGRAY),
        "tool":  (6.4, 4.4, "Act\n(Executor)", LGRAY, DGRAY),
        "world": (8.6, 3.0, "World\n(State)",  GREEN, "white"),
        "obs":   (6.4, 1.6, "Observe",          LGRAY, DGRAY),
        "check": (3.6, 1.6, "Goal\nmet?",      LGRAY, DGRAY),
    }
    for key, (x, y, label, fc, tc) in nodes.items():
        hw, hh = (0.90, 0.55) if "\n" in label else (0.82, 0.38)
        _fbox(ax, x, y, hw, hh, fc,
              ec=fc if fc in (BLUE, GREEN) else BLUE,
              lw=0 if fc in (BLUE, GREEN) else 1.0)
        _label(ax, x, y, label, tc=tc)

    g = nodes
    _arr(ax, *g["goal"][:2],  *g["plan"][:2])
    _arr(ax, *g["plan"][:2],  *g["tool"][:2])
    _arr(ax, *g["tool"][:2],  *g["world"][:2])
    _arr(ax, *g["world"][:2], *g["obs"][:2])
    _arr(ax, *g["obs"][:2],   *g["check"][:2])
    _arr(ax, *g["check"][:2], *g["plan"][:2], col=ORANGE, ls="dashed")

    ax.text(2.8, 2.50, "not yet", color=ORANGE, fontsize=9.5,
            style="italic", ha="center")
    ax.set_title("The Agentic Loop", fontsize=12.5,
                 fontweight="600", color=DGRAY, pad=8)
    plt.tight_layout(pad=0.4)
    _save(fig, save_path)
    plt.show()


def plot_tool_bench(bench: dict, save_path="images/agentic/tool_bench.png"):
    """Dual-bar chart: tool-call accuracy and latency across models.

    bench = {"model_name": {"score_pct": int, "latency_s": float}, ...}
    """
    keys   = list(bench.keys())
    labels = [short_model(k) for k in keys]
    scores = [bench[k]["score_pct"] for k in keys]
    lats   = [bench[k]["latency_s"] for k in keys]
    colors = [model_color(k) for k in keys]
    x      = np.arange(len(keys))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.0))
    fig.patch.set_facecolor("white")
    for ax in (ax1, ax2):
        _grid_ax(ax)

    for bars, vals, ax, ylabel, title, ylim, fmt in [
        (ax1.bar(x, scores, color=colors, width=0.55, zorder=3), scores, ax1,
         "accuracy (%)", "Tool-Call Accuracy", 115, "{:.0f}%"),
        (ax2.bar(x, lats,   color=colors, width=0.55, zorder=3), lats,   ax2,
         "avg latency (s)", "Latency per Call", None, "{:.1f}s"),
    ]:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9.5, color=DGRAY, rotation=15, ha="right")
        ax.set_ylabel(ylabel, fontsize=11, color=MGRAY)
        ax.set_title(title, fontsize=14, fontweight="600", color=DGRAY, pad=6)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + (1.5 if "%" in ylabel else 0.1),
                    fmt.format(v), ha="center", va="bottom",
                    fontsize=11, fontweight="600", color=DGRAY)
        if ylim:
            ax.set_ylim(0, ylim)

    fig.suptitle("Tool-Calling Model Comparison", fontsize=14,
                 fontweight="600", color=DGRAY, y=1.02)
    plt.tight_layout(pad=1.0)
    _save(fig, save_path)
    plt.show()


def plot_bakeoff(study, solid_key, faded_key, solid_label, faded_label,
                 title, ylabel, save_path, ylim=128, fmt="{:.0f}",
                 full_labels=False, solid_first=True):
    """Reusable model-zoo bake-off: per-model paired bars comparing two
    conditions. The solid bar is the headline/better condition; the faded,
    hatched bar is the baseline. Each model keeps its zoo color. By default the
    solid bar is drawn left; pass ``solid_first=False`` to draw the faded
    baseline left and the solid bar right, so a before-then-after pair reads
    left to right. With ``full_labels`` the axis keeps the ``:latest`` tag, so a
    roster that mixes size variants of one family (``gemma4:latest`` vs
    ``gemma4:e2b``) reads unambiguously.

    study = {"model": {solid_key: pct, faded_key: pct}, ...}
    """
    keys   = list(study.keys())
    labels = [k if full_labels else short_model(k) for k in keys]
    solid  = [study[k][solid_key] for k in keys]
    faded  = [study[k][faded_key] for k in keys]
    colors = [model_color(k) for k in keys]
    x = np.arange(len(keys))
    n = len(keys)
    # A big roster (the reflexion field grew to fourteen models) needs more
    # canvas, steeper labels, and smaller value text or the pairs collide. Small
    # boards (<=10) keep the original geometry, so their output is unchanged.
    crowded = n > 10
    w = 0.34 if crowded else 0.38
    fig_w = max(9.5, 0.92 * n)
    vfont = 8.0 if crowded else 9.5
    rot   = 28 if crowded else 15
    solid_off = -w/2 if solid_first else w/2     # which side each bar sits on
    faded_off = -solid_off

    fig, ax = plt.subplots(figsize=(fig_w, 4.2))
    fig.patch.set_facecolor("white")
    _grid_ax(ax)
    b1 = ax.bar(x + solid_off, solid, width=w, color=colors, zorder=3)
    b2 = ax.bar(x + faded_off, faded, width=w, color=colors, zorder=3,
                alpha=0.4, hatch="////", edgecolor="white")
    for bars, vals in [(b1, solid), (b2, faded)]:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + ylim*0.012,
                    fmt.format(v), ha="center", va="bottom", fontsize=vfont,
                    fontweight="600", color=DGRAY)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5, color=DGRAY, rotation=rot, ha="right")
    ax.set_ylabel(ylabel, fontsize=11, color=MGRAY)
    ax.set_ylim(0, ylim)
    ax.set_title(title, fontsize=14, fontweight="600", color=DGRAY, pad=10)
    sp = mp.Patch(facecolor=DGRAY, label=solid_label)
    fp = mp.Patch(facecolor=DGRAY, alpha=0.4, hatch="////", label=faded_label)
    # Sit the legend in the headroom band above the tallest bar, centered, so it
    # never lands on a bar regardless of which model is tallest. Order the legend
    # to match the on-page bar order (left entry = left bar).
    handles = [sp, fp] if solid_first else [fp, sp]
    ax.legend(handles=handles, loc="upper center", ncol=2, frameon=False,
              fontsize=10, columnspacing=1.4, handlelength=1.4)
    plt.tight_layout(pad=0.6)
    _save(fig, save_path)
    plt.show()


def plot_tool_identity(study: dict, save_path="images/agentic/tool_identity.png"):
    """Per-model tool call-rate across the four label conditions, as a heatmap.
    ``study`` is ``{model: {"clear_clear", "clear_vague", "vague_clear",
    "vague_vague"}}`` keyed name_desc. Columns run both labels clear, name only,
    description only, both vague. Rows are sorted by lean (name-only minus
    description-only), so models that read the name sit at the top and models that
    read the description at the bottom; the two middle columns are where the field
    splits, with the same crossover the showdown then shows on a single request."""
    cols    = ["clear_clear", "clear_vague", "vague_clear", "vague_vague"]
    headers = ["both\nclear", "name\nonly", "description\nonly", "both\nvague"]
    models  = sorted(study, reverse=True,
                     key=lambda m: study[m]["clear_vague"] - study[m]["vague_clear"])
    grid = np.array([[study[m][c] for c in cols] for m in models])

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    fig.patch.set_facecolor("white")
    cmap = plt.get_cmap("RdYlGn").copy()
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=100, aspect="auto")

    for i in range(len(models)):
        for j in range(len(cols)):
            v = grid[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=12,
                    fontweight="700", color="white" if v < 28 else DGRAY)

    ax.set_xticks(range(len(cols))); ax.set_xticklabels(headers, fontsize=10.5)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([short_model(m) for m in models], fontsize=10.5)
    ax.xaxis.set_label_position("top"); ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for x in (0.5, 1.5, 2.5):                       # thin gutters between columns
        ax.axvline(x, color="white", lw=2)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Which Label Does Each Model Read?  (tool call-rate, %)",
                 fontsize=12.5, fontweight="700", color=DGRAY, pad=24)
    plt.tight_layout()
    _save(fig, save_path)
    plt.show()


def plot_tool_identity_summary(study: dict,
                               save_path="images/agentic/tool_identity_summary.png"):
    """The per-model heatmap rolled into one 2x2: mean tool call-rate across models,
    tool name (clear/vague) on the rows against description (clear/vague) on the
    columns. Both-clear is hot, both-vague is the cliff, and the two off-diagonal
    cells stay warm: either label clear on its own keeps the tool alive."""
    keys = ["clear_clear", "clear_vague", "vague_clear", "vague_vague"]
    mean = {k: sum(s[k] for s in study.values()) / len(study) for k in keys}
    # rows = name (clear, vague); cols = description (clear, vague)
    grid = np.array([[mean["clear_clear"], mean["clear_vague"]],
                     [mean["vague_clear"], mean["vague_vague"]]])

    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    fig.patch.set_facecolor("white")
    cmap = plt.get_cmap("RdYlGn").copy()
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=100, aspect="auto")

    for i in range(2):
        for j in range(2):
            v = grid[i, j]
            ax.text(j, i, f"{v:.0f}%", ha="center", va="center", fontsize=30,
                    fontweight="800", color="white" if v < 28 else DGRAY)

    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["clear\n“…in the book’s glossary”",
                        "vague\n“returns a string”"], fontsize=10.5)
    ax.set_yticklabels(["clear\nlookup(term)", "vague\nfn(x)"],
                       fontsize=10.5, rotation=90, va="center")
    ax.xaxis.set_label_position("top"); ax.xaxis.tick_top()
    ax.set_xlabel("tool description", fontsize=12, fontweight="700",
                  color=DGRAY, labelpad=10)
    ax.set_ylabel("tool name", fontsize=12, fontweight="700", color=DGRAY,
                  labelpad=10)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Both Labels Matter: How Often the Model Reaches for the Tool",
                 fontsize=12.5, fontweight="700", color=DGRAY, pad=42)
    plt.tight_layout()
    _save(fig, save_path)
    plt.show()


def plot_reflexion_bakeoff(study, save_path="images/agentic/reflexion_bakeoff.png"):
    """Final gotcha-suite pass-rate, reflexion vs blind retry, per model. The gap
    up from the faded blind bar to the solid reflexion bar is the lift a model
    gets from writing itself a lesson; a flat pair means no lift (already aces it,
    or too weak to write a useful lesson)."""
    plot_bakeoff(study, "reflexion", "blind", "reflexion", "blind retry",
                 "Who Learns From Their Own Mistakes?",
                 "gotcha tasks solved (%)", save_path, solid_first=False)


def plot_confidence(study, save_path="images/agentic/confidence.png"):
    """Mean self-confidence (1-5) on answerable vs impossible questions, per model.
    The gap from the faded impossible bar up to the solid answerable bar is
    calibration: a wide gap means the model knows when it is guessing."""
    plot_bakeoff(study, "answerable", "impossible",
                 "answerable questions", "impossible questions",
                 "Who Knows What They Don't Know?",
                 "mean self-confidence (1-5)", save_path, ylim=6.4, fmt="{:.1f}")


# ── Lost-in-the-middle (reuse plot_bakeoff) ──────────────────────────────────

def plot_context(study, save_path="images/augmentation/context.png"):
    """Per model: recall of a fact at the edges of a long context vs buried in the
    middle. The drop from the solid edge bar to the faded middle bar is the
    lost-in-the-middle effect."""
    plot_bakeoff(study, "edges", "middle", "fact at the edges", "fact in the middle",
                 "Lost in the Middle?", "recall (%)", save_path)


def plot_index_speed(rows, save_path="images/augmentation/index_speed.png"):
    """Time for one search as the corpus grows: the naive linear scan (orange)
    climbs with the passage count while Chroma's HNSW index (blue) stays flat.
    Corpus sizes are spaced evenly on the x-axis so they read as a ladder."""
    sizes  = [r["n"] for r in rows]
    naive  = [r["naive_ms"] for r in rows]
    chroma = [r["chroma_ms"] for r in rows]
    x = np.arange(len(sizes))

    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    fig.patch.set_facecolor("white")
    _grid_ax(ax)
    ax.plot(x, naive, "-o", color=ORANGE, lw=2.4, ms=8, zorder=3,
            label="naive linear scan (DocumentStore)")
    ax.plot(x, chroma, "-o", color=BLUE, lw=2.4, ms=8, zorder=3,
            label="Chroma (HNSW index)")
    for xi, v in zip(x, naive):
        ax.annotate(f"{v:.0f} ms", (xi, v), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=10,
                    fontweight="600", color=ORANGE)
    ax.annotate(f"{rows[-1]['speedup']:.0f}× faster", (x[-1], chroma[-1]),
                textcoords="offset points", xytext=(-10, 18), ha="right",
                fontsize=11, fontweight="700", color=BLUE)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s:,}" for s in sizes])
    ax.set_xlabel("passages in the corpus", fontsize=11.5, color=DGRAY)
    ax.set_ylabel("time for one search (ms)", fontsize=11.5, color=DGRAY)
    ax.set_ylim(0, max(naive) * 1.2)
    ax.set_title("Linear Scan vs. Indexed Search",
                 fontsize=14, fontweight="600", color=DGRAY, pad=8)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
    plt.tight_layout(pad=0.6)
    _save(fig, save_path)
    plt.show()


def plot_scoreboard(study, value_key, title, ylabel, save_path,
                    ylim=115, fmt="{:.0f}"):
    """One bar per model on a single metric, each in its zoo color. Pass a study
    of ``{model: {value_key: number}}`` (or ``{model: number}``); the caller
    controls ordering, so sort the dict for a ranked board."""
    keys   = list(study.keys())
    labels = [short_model(k) for k in keys]
    vals   = [study[k][value_key] if isinstance(study[k], dict) else study[k]
              for k in keys]
    colors = [model_color(k) for k in keys]
    x = np.arange(len(keys))
    # A big roster (the full fourteen-model field) needs more canvas and steeper
    # labels; boards of ten or fewer keep the original geometry, so their output
    # is unchanged.
    crowded = len(keys) > 10
    fig_w = max(8.5, 0.92 * len(keys))
    rot   = 28 if crowded else 15

    fig, ax = plt.subplots(figsize=(fig_w, 4.2))
    fig.patch.set_facecolor("white")
    _grid_ax(ax)
    bars = ax.bar(x, vals, width=0.6, color=colors, zorder=3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + ylim*0.013,
                fmt.format(v), ha="center", va="bottom", fontsize=10.5,
                fontweight="600", color=DGRAY)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5, color=DGRAY, rotation=rot, ha="right")
    ax.set_ylabel(ylabel, fontsize=11, color=MGRAY)
    ax.set_ylim(0, ylim)
    ax.set_title(title, fontsize=14, fontweight="600", color=DGRAY, pad=8)
    plt.tight_layout(pad=0.6)
    _save(fig, save_path)
    plt.show()


def plot_agent_landscape(save_path="images/agentic/agent_landscape.png"):
    """Timeline of agent research milestones, one per capability theme.

    Each marker is a real, dated body of work; the years are publication dates,
    not measured quantities. This is the survey spine for the chapter intro.
    """
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")

    # (year, name, capability, theme color)
    milestones = [
        ("2022", "ReAct",              "reason + act in a loop", BLUE),
        ("2023", "Toolformer · Gorilla", "tool use and APIs",    GREEN),
        ("2023", "Generative Agents",  "human-like memory",      PURPLE),
        ("2023", "AutoGen · MetaGPT",  "teams of agents",        ORANGE),
        ("2024", "SWE-agent · Computer Use", "acting in the world", BLUE),
        ("2025", "MCP · research agents", "standardize and scale", GREEN),
    ]
    n = len(milestones)
    xs = np.linspace(0.9, 9.1, n)
    y0 = 2.5

    # baseline arrow of the timeline
    ax.annotate("", xy=(9.6, y0), xytext=(0.4, y0),
                arrowprops=dict(arrowstyle="-|>", color=MGRAY, lw=1.6,
                                mutation_scale=14, shrinkA=0, shrinkB=0))

    for i, (x, (year, name, cap, col)) in enumerate(zip(xs, milestones)):
        above = i % 2 == 0
        ly = y0 + 1.45 if above else y0 - 1.45
        # connector + dot on the timeline
        ax.plot([x, x], [y0, ly + (-0.42 if above else 0.42)],
                color=col, lw=1.2, zorder=2)
        ax.scatter([x], [y0], s=120, color=col, zorder=4,
                   edgecolors="white", linewidths=1.5)
        # label block
        _label(ax, x, ly + (0.12 if above else 0.12), name, tc=DGRAY, fs=9.5)
        ax.text(x, ly + (-0.22 if above else -0.22), cap, ha="center", va="center",
                fontsize=8, color=MGRAY, style="italic", zorder=4)
        ax.text(x + 0.34, y0 + (0.30 if above else -0.30), year, ha="left", va="center",
                fontsize=9, fontweight="700", color=col, zorder=4)

    ax.set_title("Five Years of Agent Research, One Capability at a Time",
                 fontsize=12, fontweight="600", color=DGRAY, pad=10)
    plt.tight_layout(pad=0.4)
    _save(fig, save_path)
    plt.show()


def plot_skill_retrieval(study: list, save_path="images/agentic/skill_retrieval.png"):
    """Per new task, the embedding similarity to the skill it should reuse vs the
    nearest distractor in the library. Where the right skill stands clear the
    library pays off and reuse is reliable; where a lexical neighbour crowds in,
    retrieval grabs the wrong skill (the "main idea" task lands on preview, not
    summarize). ``study`` is the row list from ``skill_retrieval_study``."""
    from matplotlib.patches import Patch
    labels = [r["label"]     for r in study]
    right  = [r["want_sim"]  for r in study]
    other  = [r["other_sim"] for r in study]
    x = np.arange(len(study)); w = 0.38

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    _grid_ax(ax)
    ax.bar(x - w/2, right, w, color=GREEN, zorder=3)
    ax.bar(x + w/2, other, w, zorder=3,
           color=[MGRAY if r["hit"] else RED for r in study])
    for xi, r, o in zip(x, right, other):
        ax.text(xi - w/2, r + 0.015, f"{r:.2f}", ha="center", va="bottom",
                fontsize=9, color=DGRAY)
        ax.text(xi + w/2, o + 0.015, f"{o:.2f}", ha="center", va="bottom",
                fontsize=9, color=DGRAY)
    ax.legend(handles=[Patch(color=GREEN, label="skill to reuse"),
                       Patch(color=MGRAY, label="nearest distractor"),
                       Patch(color=RED,   label="distractor wins (miss)")],
              fontsize=10, framealpha=0, loc="upper left")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9.5, color=DGRAY)
    ax.set_ylabel("similarity to the new task", fontsize=11, color=MGRAY)
    ax.set_ylim(0, 1.0)
    ax.set_title("A Growing Skill Library: Does the Right Skill Come Back?",
                 fontsize=13, fontweight="600", color=DGRAY, pad=8)
    plt.tight_layout(pad=0.4)
    _save(fig, save_path)
    plt.show()


# ── Augmentation chapter ──────────────────────────────────────────────────────

def plot_rag_pipeline(save_path="images/augmentation/rag_pipeline.png"):
    """Two-row RAG pipeline: indexing phase and query phase."""
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.set_xlim(0, 11); ax.set_ylim(0, 5.5); ax.axis("off")

    def box(cx, cy, label, fc, tc, hw=0.90, hh=0.50):
        _fbox(ax, cx, cy, hw, hh, fc,
              ec=fc if fc in (BLUE, GREEN) else BLUE,
              lw=0 if fc in (BLUE, GREEN) else 1.0)
        _label(ax, cx, cy, label, tc=tc)

    index_items = [
        (1.3, 3.9, "Raw\nDocs",     LGRAY, DGRAY),
        (3.2, 3.9, "Clean &\nChunk", LGRAY, DGRAY),
        (5.1, 3.9, "Embed",          LGRAY, DGRAY),
        (7.0, 3.9, "Vector\nIndex",  BLUE,  "white"),
    ]
    query_items = [
        (1.3, 1.7, "User\nQuery",    GREEN,  "white"),
        (3.2, 1.7, "Retrieve\nTop-K",LGRAY, DGRAY),
        (5.1, 1.7, "Context +\nPrompt", LGRAY, DGRAY),
        (7.0, 1.7, "Generate",        LGRAY, DGRAY),
        (9.0, 1.7, "Cited\nAnswer",  GREEN,  "white"),
    ]

    for cx, cy, lbl, fc, tc in index_items + query_items:
        box(cx, cy, lbl, fc, tc)

    for items in (index_items, query_items):
        for i in range(len(items) - 1):
            _arr(ax, items[i][0], items[i][1], items[i+1][0], items[i+1][1], pct=0.30)

    _arr(ax, 7.0, 3.9 - 0.50, 3.2, 1.7 + 0.50, col=ORANGE, pct=0.12)
    # Caption tucked into the clear gap below the diagonal arrow (the arrow runs
    # ~y3.0 across this x-span, the query boxes top out at y2.2).
    ax.text(5.7, 2.45, "retrieves from", color=ORANGE, fontsize=9.5,
            ha="center", style="italic")
    ax.text(4.15, 4.78, "Indexing Phase",
            fontsize=11, fontweight="700", color=BLUE, ha="center")
    ax.text(5.15, 0.88, "Query Phase",
            fontsize=11, fontweight="700", color=GREEN, ha="center")
    ax.set_title("Retrieval-Augmented Generation Pipeline",
                 fontsize=13, fontweight="700", color=DGRAY, pad=8)
    plt.tight_layout(pad=0.4)
    _save(fig, save_path)
    plt.show()


def plot_chunk_size(sizes: list, n_chunks: list, sims: list,
                    save_path="images/augmentation/chunk_size.png"):
    """Side-by-side bars: chunk count and top-1 similarity vs chunk size."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    fig.patch.set_facecolor("white")
    for ax in (ax1, ax2):
        _grid_ax(ax)

    labels = [str(s) for s in sizes]
    ax1.bar(labels, n_chunks, color=BLUE, width=0.5, zorder=3)
    ax1.set_xlabel("chunk size (words)", fontsize=11, color=MGRAY)
    ax1.set_ylabel("chunks produced", fontsize=11, color=MGRAY)
    ax1.set_title("Chunks Produced", fontsize=14, fontweight="600", color=DGRAY, pad=6)
    max_chunks = max(n_chunks) if n_chunks else 1
    ax1.set_ylim(0, max_chunks * 1.25)
    for i, v in enumerate(n_chunks):
        ax1.text(i, v + max_chunks * 0.04, str(v),
                 ha="center", fontsize=11, fontweight="600", color=DGRAY)

    ax2.plot(labels, sims, color=GREEN, marker="o", linewidth=2, markersize=7, zorder=3)
    ax2.set_xlabel("chunk size (words)", fontsize=11, color=MGRAY)
    ax2.set_ylabel("top-1 similarity to query", fontsize=11, color=MGRAY)
    ax2.set_title("Retrieval Quality vs. Chunk Size",
                  fontsize=14, fontweight="600", color=DGRAY, pad=6)
    ax2.set_ylim(0, 1.05)

    plt.suptitle("Chunk Size Tradeoffs", fontsize=14, fontweight="600", color=DGRAY)
    plt.tight_layout(pad=1.0)
    _save(fig, save_path)
    plt.show()


def plot_embedding_space(labels: list, vecs, colors: list,
                         group_names: list = ("Prompting", "Multimodal"),
                         title: str = "Book Passage Embeddings (PCA to 2D)",
                         save_path="images/augmentation/embedding_space.png"):
    """PCA scatter of embedding vectors, colored by group.
    Labels for nearby points are stacked vertically so they don't overlap.
    """
    from sklearn.decomposition import PCA
    from matplotlib.lines import Line2D

    coords = PCA(n_components=2).fit_transform(np.array(vecs))
    fig, ax = plt.subplots(figsize=(8, 5.2))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    for sp in ax.spines.values():
        sp.set_color(LGRAY)
    ax.tick_params(colors=MGRAY, labelsize=10)

    for (x, y), col in zip(coords, colors):
        ax.scatter(x, y, color=col, s=90, zorder=4)
    _place_labels(ax, coords, labels, fontsize=10.5)

    unique_cols = list(dict.fromkeys(colors))
    ax.legend(handles=[
        Line2D([0], [0], marker='o', color='w', markerfacecolor=c,
               ms=9, label=n)
        for c, n in zip(unique_cols, list(group_names)[:len(unique_cols)])
    ], fontsize=11, framealpha=0)
    ax.set_title(title, fontsize=14, fontweight="600", color=DGRAY, pad=8)
    plt.tight_layout(pad=0.4)
    _save(fig, save_path)
    plt.show()


def plot_retrieval_comparison(labels: list, dense: list, sparse: list, hybrid: list,
                              k: int = 3,
                              save_path="images/augmentation/retrieval_comparison.png"):
    """Grouped bar chart comparing Dense, Sparse, and Hybrid retrieval precision."""
    x = np.arange(len(labels))
    w = 0.25

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    _grid_ax(ax)

    for i, (vals, lbl, col) in enumerate([(dense, "Dense", BLUE),
                                           (sparse, "Sparse", ORANGE),
                                           (hybrid, "Hybrid", GREEN)]):
        bars = ax.bar(x + (i - 1)*w, vals, w*0.9, label=lbl, color=col, zorder=3)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width()/2, v + 0.01,
                        f"{v:.2f}", ha="center", va="bottom",
                        fontsize=10, color=DGRAY)

    # Mark groups where every retriever returned nothing with a single label.
    for xi, d, s, h in zip(x, dense, sparse, hybrid):
        if d == 0 and s == 0 and h == 0:
            ax.text(xi, 0.06, "no matches", ha="center", va="bottom",
                    fontsize=10, color=MGRAY, style="italic")

    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10, color=DGRAY)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel(f"Precision@{k}", fontsize=11, color=MGRAY)
    ax.set_title(f"Dense vs. Sparse vs. Hybrid Retrieval: Precision@{k}",
                 fontsize=14, fontweight="600", color=DGRAY, pad=8)
    ax.legend(fontsize=11, framealpha=0)
    plt.tight_layout(pad=0.4)
    _save(fig, save_path)
    plt.show()


def plot_precision_recall(k_vals: list, p_curve: list, r_curve: list,
                          save_path="images/augmentation/precision_recall.png"):
    """Precision and recall curves vs retrieval depth k."""
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    _grid_ax(ax)

    ax.plot(k_vals, p_curve, color=BLUE,  marker="o", lw=2, ms=7, label="Precision@k")
    ax.plot(k_vals, r_curve, color=GREEN, marker="s", lw=2, ms=7, label="Recall@k")

    diffs   = [abs(p - r) for p, r in zip(p_curve, r_curve)]
    ci      = diffs.index(min(diffs))
    cross_k = k_vals[ci]
    cross_y = (p_curve[ci] + r_curve[ci]) / 2
    ax.axvline(cross_k, color=ORANGE, ls="--", lw=1.2)
    ax.text(cross_k + 0.15, max(0.05, cross_y - 0.22),
            f"crossover\n(k={cross_k})", color=ORANGE, fontsize=10.5, style="italic")

    ax.set_xlabel("k (documents retrieved)", fontsize=11, color=MGRAY)
    ax.set_ylabel("score", fontsize=11, color=MGRAY)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(k_vals)
    ax.legend(fontsize=11, framealpha=0)
    ax.set_title("Precision vs. Recall Tradeoff by Retrieval Depth",
                 fontsize=14, fontweight="600", color=DGRAY, pad=8)
    plt.tight_layout(pad=0.4)
    _save(fig, save_path)
    plt.show()


def plot_rag_architecture(save_path="images/augmentation/rag_architecture.png"):
    """Three-column full RAG system architecture diagram."""
    fig, ax = plt.subplots(figsize=(9.2, 6.2))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.set_xlim(0, 11); ax.set_ylim(0, 6.5); ax.axis("off")

    SOLID = (BLUE, GREEN, ORANGE, RED)

    def lbox(cx, cy, label, fc, tc, hw=1.10, hh=0.45, sub=None):
        _fbox(ax, cx, cy, hw, hh, fc,
              ec=fc if fc in SOLID else BLUE,
              lw=0 if fc in SOLID else 0.8)
        if sub:
            _label(ax, cx, cy + 0.15, label, tc=tc, fs=10)
            ax.text(cx, cy - 0.18, sub, ha="center", va="center",
                    fontsize=8.5, color=tc if fc in SOLID else MGRAY,
                    style="italic", zorder=4)
        else:
            _label(ax, cx, cy, label, tc=tc, fs=10)

    # Column 1: Ingestion
    for y1, lbl, fc, tc in [(5.4, "Raw Docs", LGRAY, DGRAY),
                              (4.2, "Clean +\nExtract", LGRAY, DGRAY),
                              (3.0, "Chunk", LGRAY, DGRAY),
                              (1.8, "Embed", LGRAY, DGRAY),
                              (0.7, "Vector\nIndex", BLUE, "white")]:
        lbox(1.3, y1, lbl, fc, tc, hw=0.95)
    for y1, y2 in [(5.4, 4.2), (4.2, 3.0), (3.0, 1.8), (1.8, 0.7)]:
        _arr(ax, 1.3, y1 - 0.45, 1.3, y2 + 0.45, pct=0.06)

    # Column 2: Query handling
    for y1, lbl, fc, tc in [(5.4, "User Query", GREEN, "white"),
                              (4.2, "Contextualize", LGRAY, DGRAY),
                              (3.0, "Hybrid\nRetrieve", LGRAY, DGRAY),
                              (1.8, "Rerank +\nDedup", LGRAY, DGRAY),
                              (0.7, "Compress\nContext", LGRAY, DGRAY)]:
        lbox(5.5, y1, lbl, fc, tc, hw=1.05)
    for y1, y2 in [(5.4, 4.2), (4.2, 3.0), (3.0, 1.8), (1.8, 0.7)]:
        _arr(ax, 5.5, y1 - 0.45, 5.5, y2 + 0.45, pct=0.06)

    # Column 3: Generation
    for y1, lbl, fc, tc in [(3.0, "Sanitize\nInput", LGRAY, DGRAY),
                              (1.8, "Ground +\nPrompt", LGRAY, DGRAY),
                              (0.7, "Cited\nAnswer", GREEN, "white")]:
        lbox(9.4, y1, lbl, fc, tc, hw=1.05)
    for y1, y2 in [(3.0, 1.8), (1.8, 0.7)]:
        _arr(ax, 9.4, y1 - 0.45, 9.4, y2 + 0.45, pct=0.08)

    _arr(ax, 1.3 + 0.95, 0.7, 5.5 - 1.05, 3.0, col=ORANGE, pct=0.08)
    _arr(ax, 5.5 + 1.05, 0.7, 9.4 - 1.05, 1.8, col=ORANGE, pct=0.10)
    _arr(ax, 5.5 + 1.05, 5.4, 9.4 - 1.05, 3.0, col=MGRAY, pct=0.10, ls="dashed")

    for x, title in [(1.3, "Indexing"), (5.5, "Retrieval"), (9.4, "Generation")]:
        ax.text(x, 6.1, title, ha="center", fontsize=11.5, fontweight="700", color=DGRAY)
        ax.plot([x - 1.1, x + 1.1], [5.85, 5.85], color=LGRAY, lw=1.2)

    ax.set_title("Full RAG System Architecture",
                 fontsize=13.5, fontweight="700", color=DGRAY, pad=10)
    plt.tight_layout(pad=0.4)
    _save(fig, save_path)
    plt.show()


# ── Metacoding chapter ────────────────────────────────────────────────────────

def render_code_listing(snippets, font_size=30, pad=24, scale=2, save_path=None):
    """Render Python snippets as a syntax-highlighted image.

    Lets a chapter *show* a small code dataset as a figure (Pygments ``friendly``
    style — the same one the PDF's ``minted`` uses — set in IBM Plex Mono, the
    book's code font) instead of pasting it inline as string literals. The real
    snippets live as named constants in ``genai.code``; this just paints them.

    ``snippets`` is a single code string, or a list whose items are code strings
    or ``(name, code)`` pairs (the name is ignored — the ``def`` line already
    carries it). Multi-line snippets are spaced apart by a blank line. Returns a
    PIL image, which Jupyter displays as the cell's figure.
    """
    from pathlib import Path
    from PIL import Image, ImageDraw, ImageFont
    from pygments.lexers import PythonLexer
    from pygments.styles import get_style_by_name

    if isinstance(snippets, str):
        blocks = [snippets]
    else:
        blocks = [s[1] if isinstance(s, (tuple, list)) else s for s in snippets]
    sep = "\n\n" if any("\n" in b for b in blocks) else "\n"
    code = sep.join(b.rstrip("\n") for b in blocks)

    fonts = Path(__file__).resolve().parents[2] / "fonts"
    fs, p = font_size * scale, pad * scale
    reg = ImageFont.truetype(str(fonts / "IBMPlexMono-Regular.ttf"), fs)
    bold = ImageFont.truetype(str(fonts / "IBMPlexMono-Bold.ttf"), fs)
    ital = ImageFont.truetype(str(fonts / "IBMPlexMono-Italic.ttf"), fs)
    cw = reg.getlength("M")                       # monospace cell width
    asc, desc = reg.getmetrics()
    lh = int((asc + desc) * 1.34)                 # line height

    style = get_style_by_name("friendly")
    rows = code.split("\n")
    W = int(p * 2 + cw * max((len(r) for r in rows), default=1))
    H = int(p * 2 + lh * len(rows))
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)

    x, y = p, p
    for ttype, value in PythonLexer().get_tokens(code):
        st = style.style_for_token(ttype)
        color = "#" + st["color"] if st["color"] else DGRAY
        font = bold if st["bold"] else (ital if st["italic"] else reg)
        for i, part in enumerate(value.split("\n")):
            if i:                                 # token spanned a line break
                x, y = p, y + lh
            if part:
                draw.text((x, y), part, font=font, fill=color)
                x += cw * len(part)

    img = img.resize((W // scale, H // scale), Image.LANCZOS)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        img.save(save_path)
    return img


def render_code_columns(columns, gap=44, save_path=None, **kwargs):
    """Render several code blocks as side-by-side syntax-highlighted columns.

    ``columns`` is a list; each item is anything ``render_code_listing`` accepts.
    Use this when a single listing would be too tall for the page. Columns are
    top-aligned on a white canvas with a ``gap`` of whitespace between them.
    """
    from PIL import Image
    imgs = [render_code_listing(col, **kwargs) for col in columns]
    width = sum(im.width for im in imgs) + gap * (len(imgs) - 1)
    height = max(im.height for im in imgs)
    canvas = Image.new("RGB", (width, height), "white")
    x = 0
    for im in imgs:
        canvas.paste(im, (x, 0))
        x += im.width + gap
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        canvas.save(save_path)
    return canvas


def plot_code_embeddings(save_path="images/metacoding/embeddings_2d.png"):
    """PCA projection of code function embeddings, colored by category."""
    from sklearn.decomposition import PCA
    from genai.code import embed_code

    CATEGORIES = {
        "arithmetic": [
            ("add",       "def add(a, b): return a + b"),
            ("multiply",  "def multiply(a, b): return a * b"),
            ("power",     "def power(base, exp): return base ** exp"),
            ("factorial", "def factorial(n): return 1 if n == 0 else n * factorial(n-1)"),
        ],
        "strings": [
            ("reverse",    "def reverse(s): return s[::-1]"),
            ("uppercase",  "def uppercase(s): return s.upper()"),
            ("palindrome", "def is_palindrome(s): return s == s[::-1]"),
            ("word_count", "def word_count(s): return len(s.split())"),
        ],
        "lists": [
            ("find_max",  "def find_max(lst): return max(lst)"),
            ("sort_desc", "def sort_desc(lst): return sorted(lst, reverse=True)"),
            ("flatten",   "def flatten(lst): return [x for sub in lst for x in sub]"),
            ("evens",     "def evens(lst): return [x for x in lst if x % 2 == 0]"),
        ],
        "file I/O": [
            ("read_file",  "def read_file(path): return open(path).read()"),
            ("write_file", "def write_file(path, data): open(path, 'w').write(data)"),
            ("log_error",  "def log_error(msg): open('error.log','a').write(msg + '\\n')"),
        ],
    }
    COLORS = {"arithmetic": BLUE, "strings": GREEN, "lists": ORANGE, "file I/O": PURPLE}

    vecs, labels, colors, cats = [], [], [], []
    for cat, items in CATEGORIES.items():
        for lbl, code in items:
            vecs.append(embed_code(code))
            labels.append(lbl)
            colors.append(COLORS[cat])
            cats.append(cat)

    coords = PCA(n_components=2).fit_transform(np.stack(vecs))

    fig, ax = plt.subplots(figsize=(9, 5.8))
    for cat in CATEGORIES:
        idx = [i for i, c in enumerate(cats) if c == cat]
        ax.scatter(coords[idx, 0], coords[idx, 1],
                   c=COLORS[cat], s=100, label=cat, zorder=3,
                   edgecolors="white", lw=0.8)
    _place_labels(ax, coords, labels, fontsize=10.5)

    ax.set_title("Code Embeddings in 2D  (15 functions, PCA projection)",
                 fontsize=14, color=DGRAY, pad=12)
    ax.set_xlabel("First principal component", fontsize=11, color=MGRAY)
    ax.set_ylabel("Second principal component", fontsize=11, color=MGRAY)
    ax.legend(fontsize=10.5, loc="upper left", framealpha=0.9, title="category")
    ax.grid(alpha=0.15)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=10, colors=MGRAY)
    plt.tight_layout()
    _save(fig, save_path)
    plt.show()


def plot_search_comparison(search_results: list,
                           save_path="images/metacoding/model_comparison.png"):
    """Grouped bar chart comparing two embedding models on code search results.

    search_results = [(name, unixcoder_score, nomic_score), ...]
    """
    names  = [r[0] for r in search_results]
    uni_sc = [r[1] for r in search_results]
    nom_sc = [r[2] for r in search_results]
    x = np.arange(len(names))
    w = 0.38

    fig, ax = plt.subplots(figsize=(10, 4.4))
    ax.bar(x - w/2, uni_sc, w, color=BLUE,  label="UniXcoder",       zorder=2)
    ax.bar(x + w/2, nom_sc, w, color=GREEN, label="nomic-embed-text", zorder=2)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9.5)
    ax.set_ylabel("Similarity to query", fontsize=12.5, color=DGRAY)
    ax.set_title('Search scores for "combine two numbers"',
                 fontsize=14, color=DGRAY, pad=10)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(axis="y", alpha=0.2, zorder=1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=11)
    plt.tight_layout()
    _save(fig, save_path)
    plt.show()


def plot_embedder_comparison(results,
                             save_path="images/metacoding/embedder_comparison.png"):
    """Grouped bar chart comparing three code embedders on one search query.

    results = [(name, unixcoder, codebert, codesearch), ...]. The y-axis dips
    below zero because CodeSearch can score an irrelevant function negative.
    """
    names = [r[0] for r in results]
    uni   = [r[1] for r in results]
    cb    = [r[2] for r in results]
    csn   = [r[3] for r in results]
    x = np.arange(len(names))
    w = 0.27

    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.bar(x - w, uni, w, color=BLUE,  label="UniXcoder",  zorder=2)
    ax.bar(x,     cb,  w, color=AMBER, label="CodeBERT",   zorder=2)
    ax.bar(x + w, csn, w, color=TEAL,  label="CodeSearch", zorder=2)
    ax.axhline(0, color=MGRAY, lw=0.9, zorder=3)

    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=10.5)
    ax.set_ylabel("Similarity to query", fontsize=12.5, color=DGRAY)
    ax.set_title('Three code embedders score "combine two numbers"',
                 fontsize=14, color=DGRAY, pad=10)
    ax.set_ylim(-0.2, 1.12)
    ax.legend(fontsize=10.5, ncol=3, loc="upper center", framealpha=0.9)
    ax.grid(axis="y", alpha=0.2, zorder=1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=11)
    plt.tight_layout()
    _save(fig, save_path)
    plt.show()


def plot_model_speed(bench: list, save_path="images/metacoding/speed.png"):
    """Horizontal bar chart of tokens/sec for each model.

    bench = [{"label": str, "tps": float, "params_b": float}, ...]
    Color bins are only shown in the legend if they appear in the data.
    """
    labels = [b["label"] for b in bench]
    tps    = [b["tps"]   for b in bench]

    def bin_for(t):
        if t > 60: return ("fast",   BLUE,   "> 60 tok/s")
        if t > 35: return ("medium", GREEN,  "35–60 tok/s")
        return         ("slow",   ORANGE, "< 35 tok/s")

    bin_info = [bin_for(t) for t in tps]
    colors = [c for _, c, _ in bin_info]

    import matplotlib.patches as mpatches
    fig, ax = plt.subplots(figsize=(9, 4.2))
    bars = ax.barh(labels, tps, color=colors, height=0.52, zorder=2)
    for bar, val in zip(bars, tps):
        ax.text(val + 1.8, bar.get_y() + bar.get_height()/2,
                f"{val:.0f} tok/s", va="center", fontsize=11, color=DGRAY)

    ax.set_xlabel("Tokens per second", fontsize=12.5, color=DGRAY)
    # Extra right-side headroom so value labels never collide with the legend.
    ax.set_xlim(0, max(tps) * 1.35)
    ax.grid(axis="x", alpha=0.2, zorder=1)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="both", labelsize=11)
    ax.set_title("Coding Model Speed (local, Apple Silicon)",
                 fontsize=14, fontweight="700", color=DGRAY, pad=10)

    seen = {}
    for name, col, lbl in bin_info:
        seen.setdefault(name, (col, lbl))
    if len(seen) > 1:
        ax.legend(handles=[mpatches.Patch(color=c, label=l) for c, l in seen.values()],
                  loc="upper right",
                  bbox_to_anchor=(1.0, 1.0),
                  fontsize=10, framealpha=0.9)
    plt.tight_layout()
    _save(fig, save_path)
    plt.show()


def plot_size_vs_speed(bench: list, save_path="images/metacoding/size_vs_speed.png"):
    """Scatter: model parameter count vs tokens/sec.

    bench = [{"label": str, "tps": float, "params_b": float}, ...]
    Labels for points within a small neighborhood get stacked vertically
    so they don't collide.
    """
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for b in bench:
        ax.scatter(b["params_b"], b["tps"], s=140, color=BLUE, zorder=3,
                   edgecolors="white", linewidths=0.8)

    tps_range = max(b["tps"] for b in bench) - min(b["tps"] for b in bench) or 1.0
    par_range = max(b["params_b"] for b in bench) - min(b["params_b"] for b in bench) or 1.0

    for i, b in enumerate(bench):
        # Count how many earlier points are in this point's cluster.
        stack_idx = sum(
            1 for j, other in enumerate(bench)
            if j < i
            and abs(b["params_b"] - other["params_b"]) < par_range * 0.10
            and abs(b["tps"] - other["tps"]) < tps_range * 0.04
        )
        ax.annotate(b["label"],
                    (b["params_b"], b["tps"]),
                    xytext=(10, 4 + stack_idx * 14),
                    textcoords="offset points",
                    fontsize=11, color=DGRAY, va="center")

    ax.set_xlabel("Model size (billions of parameters)", fontsize=12.5, color=DGRAY)
    ax.set_ylabel("Tokens per second", fontsize=12.5, color=DGRAY)
    ax.set_title("Smaller Models Generate Faster",
                 fontsize=14, fontweight="700", color=DGRAY, pad=10)
    ax.grid(alpha=0.18, zorder=1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=11)
    # A touch of right-padding so labels don't run off-canvas.
    xlim = ax.get_xlim()
    ax.set_xlim(xlim[0], xlim[1] + (xlim[1] - xlim[0]) * 0.05)
    plt.tight_layout()
    _save(fig, save_path)
    plt.show()


# ── Efficiency chapter ────────────────────────────────────────────────────────

def _short(model: str) -> str:
    """Trim the ':latest' / 'instruct' noise off a model tag for axis labels."""
    return model.replace(":latest", "").replace("-instruct", "")


def plot_throughput(rows: list, save_path="images/efficiency/throughput.png"):
    """Two panels on one short question across models: per-token speed (left)
    and total time to answer split into prompt-reading vs generation (right).

    rows = [{"model": str, "tokens_per_sec"|"tps": float, "prompt_ms": float,
             "gen_tokens": int}, ...]
    The pairing reveals two things at once: the fastest model per token is not
    the fastest to finish, and almost the whole wait is generation. The
    prompt-reading sliver barely registers while a reasoning model that thinks
    for thousands of tokens balloons the generation slab. Generation time is
    derived as gen_tokens / tokens_per_sec, the pure writing time, so a one-off
    model load never leaks into the slab.
    """
    labels = [_short(r["model"]) for r in rows]
    tps    = [r.get("tps", r.get("tokens_per_sec")) for r in rows]
    toks   = [r.get("gen_tokens", 0) for r in rows]
    read_s = [r["prompt_ms"] / 1000 for r in rows]
    gen_s  = [t / v if v else 0.0 for t, v in zip(toks, tps)]
    secs   = [r + g for r, g in zip(read_s, gen_s)]
    x = np.arange(len(rows))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.6))
    fig.patch.set_facecolor("white")
    for ax in (ax1, ax2):
        _grid_ax(ax)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9.5, color=DGRAY, rotation=18, ha="right")

    bars1 = ax1.bar(x, tps, color=GREEN, width=0.55, zorder=3)
    ax1.set_ylabel("tokens / sec", fontsize=11, color=MGRAY)
    ax1.set_title("Speed per Token", fontsize=13, fontweight="600", color=DGRAY, pad=6)
    ax1.set_ylim(0, max(tps) * 1.18)
    for bar, v in zip(bars1, tps):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(tps)*0.02,
                 f"{v:.0f}", ha="center", va="bottom",
                 fontsize=10.5, fontweight="600", color=DGRAY)

    ax2.bar(x, read_s, color=BLUE, width=0.55, zorder=3, label="reading")
    ax2.bar(x, gen_s, bottom=read_s, color=ORANGE, width=0.55, zorder=3,
            label="generation")
    ax2.set_ylabel("time to answer (s)", fontsize=11, color=MGRAY)
    ax2.set_title("Reading vs Generation Time", fontsize=13, fontweight="600",
                  color=DGRAY, pad=6)
    ax2.set_ylim(0, max(secs) * 1.22)
    ax2.legend(fontsize=9.5, frameon=False, loc="upper left")
    for xi, s, t in zip(x, secs, toks):
        ax2.text(xi, s + max(secs)*0.02, f"{s:.0f}s\n{t} tok", ha="center",
                 va="bottom", fontsize=9.5, fontweight="600", color=DGRAY,
                 linespacing=1.1)

    fig.suptitle("Same One-Sentence Question, Four Models", fontsize=14,
                 fontweight="600", color=DGRAY, y=1.04)
    plt.tight_layout(pad=1.0)
    _save(fig, save_path)
    plt.show()


def plot_context_cost(rows: list, save_path="images/efficiency/context_cost.png"):
    """Line chart: prompt-processing time grows linearly with prompt length.

    rows = [{"prompt_tokens": int, "prompt_ms": float}, ...]
    """
    toks = [r["prompt_tokens"] for r in rows]
    ms   = [r["prompt_ms"] for r in rows]

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    _grid_ax(ax)
    ax.plot(toks, ms, color=BLUE, marker="o", lw=2, ms=8, zorder=3)
    for t, m in zip(toks, ms):
        ax.text(t, m + max(ms) * 0.03, f"{m:.0f}", ha="center",
                fontsize=10, color=DGRAY)
    ax.set_xlabel("prompt length (tokens)", fontsize=11.5, color=MGRAY)
    ax.set_ylabel("prompt-processing time (ms)", fontsize=11.5, color=MGRAY)
    ax.set_ylim(0, max(ms) * 1.18)
    ax.set_title("Every Extra Token in the Prompt Costs Time",
                 fontsize=14, fontweight="600", color=DGRAY, pad=8)
    plt.tight_layout(pad=0.6)
    _save(fig, save_path)
    plt.show()


def plot_conversation_cache(warm: list, cold: list,
                            save_path="images/efficiency/conversation_cache.png"):
    """Two panels: transcript size per turn, and reading time warm vs cold.

    warm/cold = per-turn timing dicts from time_chat. The transcript grows
    every turn; a fresh model pays the full reading bill while the ongoing
    chat, served from the KV cache, reads only the newest question.
    """
    turns = np.arange(1, len(cold) + 1)
    toks = [r["prompt_tokens"] for r in cold]
    cms = [r["prompt_ms"] for r in cold]
    wms = [r["prompt_ms"] for r in warm]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.6))
    fig.patch.set_facecolor("white")
    for ax in (ax1, ax2):
        ax.set_facecolor("white")
        _grid_ax(ax)
        ax.set_xticks(turns)
        ax.set_xlabel("turn", fontsize=11, color=MGRAY)

    ax1.bar(turns, toks, color=BLUE, width=0.6, zorder=3)
    for t, v in zip(turns, toks):
        ax1.text(t, v + max(toks) * 0.03, f"{v}", ha="center",
                 fontsize=10, fontweight="600", color=DGRAY)
    ax1.set_ylim(0, max(toks) * 1.18)
    ax1.set_ylabel("transcript (tokens)", fontsize=11, color=MGRAY)
    ax1.set_title("What the App Sends Each Turn", fontsize=13,
                  fontweight="600", color=DGRAY, pad=6)

    ax2.plot(turns, cms, color=RED, marker="o", lw=2, ms=7, zorder=3,
             label="fresh model (no cache)")
    ax2.plot(turns, wms, color=GREEN, marker="o", lw=2, ms=7, zorder=3,
             label="ongoing chat (cached)")
    for t, v in zip(turns, cms):
        ax2.text(t, v + max(cms) * 0.04, f"{v:.0f}", ha="center",
                 fontsize=9.5, color=DGRAY)
    ax2.text(turns[-1], wms[-1] + max(cms) * 0.04, f"{wms[-1]:.0f}",
             ha="center", fontsize=9.5, color=DGRAY)
    ax2.set_ylim(0, max(cms) * 1.22)
    ax2.set_ylabel("reading time (ms)", fontsize=11, color=MGRAY)
    ax2.legend(fontsize=10, frameon=False, loc="upper left")
    ax2.set_title("Time the Model Spends Reading", fontsize=13,
                  fontweight="600", color=DGRAY, pad=6)

    fig.suptitle("Six Turns, One Growing Transcript", fontsize=14,
                 fontweight="600", color=DGRAY, y=1.04)
    plt.tight_layout(pad=1.0)
    _save(fig, save_path)
    plt.show()


def plot_generation_sag(rows: list,
                        save_path="images/efficiency/generation_sag.png"):
    """Line: generation speed vs transcript length across one long chat.

    rows = per-turn timing dicts from time_chat. The KV cache spares the
    re-reading, but every new token still attends over the whole cache,
    so tokens/sec sags as the transcript grows.
    """
    toks = [r["prompt_tokens"] for r in rows]
    tps = [r["tokens_per_sec"] for r in rows]

    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    _grid_ax(ax)
    ax.plot(toks, tps, color=ORANGE, marker="o", lw=2, ms=6, zorder=3)
    for i in (0, len(rows) - 1):
        ax.annotate(f"{tps[i]:.1f}", (toks[i], tps[i]),
                    textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=10.5, fontweight="600", color=DGRAY)
    ax.set_xlabel("transcript (tokens)", fontsize=11.5, color=MGRAY)
    ax.set_ylabel("generation speed (tokens / sec)", fontsize=11.5, color=MGRAY)
    ax.set_ylim(0, max(tps) * 1.18)
    ax.set_title("Writing Slows as the Transcript Grows",
                 fontsize=14, fontweight="600", color=DGRAY, pad=8)
    plt.tight_layout(pad=0.6)
    _save(fig, save_path)
    plt.show()


def plot_prompt_compression(rows: list,
                            save_path="images/efficiency/prompt_compression.png"):
    """Paired horizontal bars: prompt size and reading time per phrasing.

    rows = [{"label": str, "prompt_tokens": int, "prompt_ms": float}, ...]
    The same question dressed three ways; the padding, not the question,
    sets the cost. Rows are sorted largest-first so the most padded
    phrasing lands on top.
    """
    rows = sorted(rows, key=lambda r: -r["prompt_tokens"])
    labels = [r["label"] for r in rows]
    toks = [r["prompt_tokens"] for r in rows]
    ms = [r["prompt_ms"] for r in rows]
    colors = [RED, ORANGE, GREEN, BLUE, PURPLE][:len(rows)]
    y = np.arange(len(rows))[::-1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.0), sharey=True)
    fig.patch.set_facecolor("white")
    for ax, vals, unit, title in [
            (ax1, toks, "tokens", "What the Model Reads"),
            (ax2, ms, "milliseconds", "Time Before Response")]:
        _grid_ax(ax)
        ax.xaxis.grid(True, color=LGRAY, zorder=0)
        ax.yaxis.grid(False)
        ax.barh(y, vals, color=colors, height=0.58, zorder=3)
        for yi, v in zip(y, vals):
            ax.text(v + max(vals) * 0.02, yi, f"{v:.0f}", va="center",
                    fontsize=10.5, fontweight="600", color=DGRAY)
        ax.set_xlim(0, max(vals) * 1.16)
        ax.set_xlabel(unit, fontsize=11, color=MGRAY)
        ax.set_title(title, fontsize=13, fontweight="600", color=DGRAY, pad=6)
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels, fontsize=11, color=DGRAY)
    fig.suptitle("Price of Politeness", fontsize=14, fontweight="600",
                 color=DGRAY, y=1.04)
    plt.tight_layout(pad=1.0)
    _save(fig, save_path)
    plt.show()


def plot_task_ladder(rows: list, save_path="images/efficiency/task_ladder.png"):
    """Grouped bars: generation speed for a small vs large model up a task ladder.

    rows = [{"task": str, "small": float, "large": float}, ...]
    The flat tops carry the lesson: per-token speed barely moves as the task
    gets harder, because speed is a property of the model, not the question.
    """
    tasks = [r["task"] for r in rows]
    small = [r["small"] for r in rows]
    large = [r["large"] for r in rows]
    x = np.arange(len(rows))
    w = 0.36

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    _grid_ax(ax)
    top = max(small + large)
    for offs, vals, color, label in [(-w/2, small, GREEN, "1B model"),
                                     (+w/2, large, BLUE, "3B model")]:
        bars = ax.bar(x + offs, vals, w, color=color, zorder=3, label=label)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + top*0.02,
                    f"{v:.0f}", ha="center", va="bottom",
                    fontsize=10.5, fontweight="600", color=DGRAY)
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=11.5, color=DGRAY)
    ax.set_ylabel("tokens / sec", fontsize=11, color=MGRAY)
    ax.set_ylim(0, top * 1.38)
    ax.legend(fontsize=10.5, frameon=False, loc="upper center", ncols=2)
    ax.set_title("Per-Token Speed Belongs to the Model, Not the Task",
                 fontsize=13.5, fontweight="600", color=DGRAY, pad=8)
    plt.tight_layout(pad=0.6)
    _save(fig, save_path)
    plt.show()


def plot_quantization(rows: list, save_path="images/efficiency/quantization.png"):
    """Dual bars: disk footprint and generation speed across quantization levels.

    rows = [{"model": str, "size_gb": float, "tps": float}, ...]
    """
    labels = [_short(r["model"]).split(":")[-1] for r in rows]
    size   = [r["size_gb"] for r in rows]
    tps    = [r["tps"] for r in rows]
    colors = [GREEN, BLUE, ORANGE][:len(rows)]
    x = np.arange(len(rows))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.6))
    fig.patch.set_facecolor("white")
    for ax in (ax1, ax2):
        _grid_ax(ax)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10, color=DGRAY)

    for ax, vals, title, ylab, fmt in [
        (ax1, size, "Disk Footprint", "gigabytes", "{:.2f}"),
        (ax2, tps,  "Generation Speed", "tokens / sec", "{:.0f}"),
    ]:
        bars = ax.bar(x, vals, color=colors, width=0.55, zorder=3)
        ax.set_ylabel(ylab, fontsize=11, color=MGRAY)
        ax.set_title(title, fontsize=13, fontweight="600", color=DGRAY, pad=6)
        ax.set_ylim(0, max(vals) * 1.18)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.02,
                    fmt.format(v), ha="center", va="bottom",
                    fontsize=10.5, fontweight="600", color=DGRAY)
    fig.suptitle("One Model, Three Quantizations (Llama 3.2 1B)",
                 fontsize=13.5, fontweight="600", color=DGRAY, y=1.03)
    plt.tight_layout(pad=1.0)
    _save(fig, save_path)
    plt.show()


# Active-parameter colors: dense pays for everything, a mixture-of-experts
# model lights up only a sliver, and an on-device model sits between the two.
_SPARSITY_COLOR = {"dense": BLUE, "selective": TEAL, "expert": ORANGE}


def _sparsity_kind(kind: str) -> str:
    """Bucket a model's description into dense / selective / expert."""
    if "expert" in kind:
        return "expert"
    if "selective" in kind or "PLE" in kind:
        return "selective"
    return "dense"


def plot_sparsity(rows: list, save_path="images/efficiency/sparsity.png"):
    """Two panels: parameters spent per token, and generation speed.

    rows = [{"model": str, "total_b": float, "active_b": float, "tps": float,
             "kind": str}, ...]
    The left panel draws each model's stored parameters as a light bar with the
    parameters it actually activates per token filled solid on top, so a
    mixture-of-experts model shows a tall light bar over a short solid core.
    The right panel shows decode speed tracks that active core, not the stored
    total: the MoE model decodes as fast as dense models a fraction of its size.
    """
    import matplotlib.patches as mpatches
    labels = [_short(r["model"]).split(":")[0] for r in rows]
    total  = [r["total_b"] for r in rows]
    active = [r["active_b"] for r in rows]
    tps    = [r["tps"] for r in rows]
    colors = [_SPARSITY_COLOR[_sparsity_kind(r["kind"])] for r in rows]
    x = np.arange(len(rows))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.9))
    fig.patch.set_facecolor("white")
    for ax in (ax1, ax2):
        _grid_ax(ax)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9.5, color=DGRAY)

    # Left: stored parameters (light) with the active core (solid) on top.
    ax1.bar(x, total, width=0.62, color=LGRAY, zorder=2)
    ax1.bar(x, active, width=0.62, color=colors, zorder=3)
    ax1.set_ylabel("parameters (billions)", fontsize=11, color=MGRAY)
    ax1.set_title("Stored vs Active per Token", fontsize=13, fontweight="600",
                  color=DGRAY, pad=6)
    ax1.set_ylim(0, max(total) * 1.20)
    for xi, t, a in zip(x, total, active):
        ax1.text(xi, t + max(total) * 0.02, f"{t:.1f}", ha="center", va="bottom",
                 fontsize=9.5, color=MGRAY)
        if a < t * 0.85:   # label the active core only when it is visibly smaller
            ax1.text(xi, a + max(total) * 0.02, f"{a:.0f}", ha="center",
                     va="bottom", fontsize=9.5, fontweight="700", color=DGRAY)
    present = [k for k in ("dense", "selective", "expert")
               if k in {_sparsity_kind(r["kind"]) for r in rows}]
    legend_label = {"dense": "active (dense)", "selective": "active (on-device)",
                    "expert": "active (mixture-of-experts)"}
    handles = [mpatches.Patch(color=LGRAY, label="stored")] + \
              [mpatches.Patch(color=_SPARSITY_COLOR[k], label=legend_label[k])
               for k in present]
    ax1.legend(handles=handles, fontsize=8.5, frameon=False, loc="upper left",
               handlelength=1.1, labelspacing=0.3)

    # Right: decode speed, colored to match the active core.
    bars = ax2.bar(x, tps, width=0.58, color=colors, zorder=3)
    ax2.set_ylabel("tokens / sec", fontsize=11, color=MGRAY)
    ax2.set_title("Generation Speed", fontsize=13, fontweight="600",
                  color=DGRAY, pad=6)
    ax2.set_ylim(0, max(tps) * 1.20)
    for bar, v in zip(bars, tps):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(tps)*0.02,
                 f"{v:.0f}", ha="center", va="bottom",
                 fontsize=10.5, fontweight="600", color=DGRAY)

    fig.suptitle("Store a Big Model, Run a Small One",
                 fontsize=14, fontweight="600", color=DGRAY, y=1.04)
    plt.tight_layout(pad=1.0)
    _save(fig, save_path)
    plt.show()


def plot_effort_cost(study: dict, save_path="images/efficiency/effort_cost.png"):
    """Two panels: what a reasoning-effort dial costs vs what it buys.

    study = {"efforts": [...], "bill": [...], "gen_s": [...], "correct": [...],
             "n_tasks": int}
    Left, the mean token bill per effort level, rising green to red with the
    answer time labelled on each bar. Right, accuracy over the same tasks, which
    stays flat: turning the dial up spends more and returns the same answers.
    """
    efforts = [e.capitalize() for e in study["efforts"]]
    bill = study["bill"]
    gen_s = study["gen_s"]
    n = study["n_tasks"]
    acc = [100.0 * c / n for c in study["correct"]]
    x = np.arange(len(efforts))
    cost_colors = [GREEN, AMBER, RED][:len(efforts)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.7))
    fig.patch.set_facecolor("white")
    for ax in (ax1, ax2):
        _grid_ax(ax)
        ax.set_xticks(x)
        ax.set_xticklabels(efforts, fontsize=10.5, color=DGRAY)
        ax.set_xlabel("reasoning effort", fontsize=10.5, color=MGRAY)

    # Left: token bill (rising), with answer time labelled on each bar.
    bars = ax1.bar(x, bill, width=0.6, color=cost_colors, zorder=3)
    ax1.set_ylabel("tokens per answer", fontsize=11, color=MGRAY)
    ax1.set_title("What You Pay", fontsize=13, fontweight="600", color=DGRAY, pad=6)
    ax1.set_ylim(0, max(bill) * 1.22)
    for bar, b, s in zip(bars, bill, gen_s):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(bill)*0.02,
                 f"{b} tok\n{s:.1f}s", ha="center", va="bottom",
                 fontsize=10, fontweight="600", color=DGRAY, linespacing=1.1)

    # Right: accuracy (flat) over the same tasks.
    abars = ax2.bar(x, acc, width=0.6, color=GREEN, zorder=3)
    ax2.set_ylabel("tasks correct (%)", fontsize=11, color=MGRAY)
    ax2.set_title("What You Get", fontsize=13, fontweight="600", color=DGRAY, pad=6)
    ax2.set_ylim(0, 119)
    for bar, c in zip(abars, study["correct"]):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                 f"{c}/{n}", ha="center", va="bottom",
                 fontsize=10.5, fontweight="600", color=DGRAY)

    fig.suptitle("The Price of Thinking", fontsize=14, fontweight="600",
                 color=DGRAY, y=1.04)
    plt.tight_layout(pad=1.0)
    _save(fig, save_path)
    plt.show()


def plot_embed_vs_gen(n: int, embed_s: float, gen_s: float,
                      save_path="images/efficiency/embed_vs_gen.png"):
    """Two bars on a log axis: embedding vs generation throughput (items/sec)."""
    embed_rate = n / embed_s
    gen_rate   = n / gen_s
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    _grid_ax(ax)
    bars = ax.bar(["embedding", "generation"], [embed_rate, gen_rate],
                  color=[GREEN, ORANGE], width=0.5, zorder=3)
    ax.set_yscale("log")
    ax.set_ylabel("items per second (log scale)", fontsize=11.5, color=MGRAY)
    for bar, v in zip(bars, [embed_rate, gen_rate]):
        ax.text(bar.get_x() + bar.get_width()/2, v * 1.12,
                f"{v:.1f}/s", ha="center", va="bottom",
                fontsize=11, fontweight="600", color=DGRAY)
    ax.set_title(f"Embedding Is ~{embed_rate/gen_rate:.0f}x Faster Than Generation",
                 fontsize=13.5, fontweight="600", color=DGRAY, pad=8)
    plt.tight_layout(pad=0.6)
    _save(fig, save_path)
    plt.show()


# ── Responsible chapter ───────────────────────────────────────────────────────

def plot_privacy_budget(rows, save_path="images/responsible/privacy_budget.png"):
    """The differential-privacy tradeoff curve: error in the released statistic
    against the privacy budget epsilon.

    rows = [(epsilon, private_mean, abs_error, strength), ...] as returned by
    genai.security.dp_budget_table. The error collapses as epsilon grows, which
    is precisely the privacy you spend to buy that accuracy back.
    """
    eps = [r[0] for r in rows]
    err = [r[2] for r in rows]
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    _grid_ax(ax)
    ax.set_xscale("log")
    ax.plot(eps, err, color=RED, marker="o", lw=2.2, ms=9, zorder=3)
    ax.fill_between(eps, err, color=RED, alpha=0.08, zorder=1)
    for e, v in zip(eps, err):
        ax.text(e, v + max(err) * 0.04, f"{v:.1f}", ha="center",
                fontsize=10, color=DGRAY, zorder=4)
    ax.set_xlabel("privacy budget  ε   (log scale)", fontsize=11.5, color=MGRAY)
    ax.set_ylabel("error in released average (mmHg)", fontsize=11.5, color=MGRAY)
    ax.set_ylim(0, max(err) * 1.25)
    ax.text(eps[0], max(err) * 1.12, "strong privacy\nlots of noise", fontsize=9.5,
            color=GREEN, ha="left", va="top", fontweight="700")
    ax.text(eps[-1], max(err) * 0.5, "weak privacy\nlittle noise", fontsize=9.5,
            color=ORANGE, ha="right", va="center", fontweight="700")
    ax.set_title("The Privacy Budget: No Free Lunch", fontsize=14,
                 fontweight="700", color=DGRAY, pad=8)
    plt.tight_layout(pad=0.6)
    _save(fig, save_path)
    plt.show()


def plot_watermark_detection(watermarked_z, human_z,
                             save_path="images/responsible/watermark_detection.png"):
    """Strip plot of green-list z-scores for watermarked vs. human text.

    Human text scatters around zero (about half its tokens are 'green' by
    chance); watermarked text spikes well past the usual z=4 detection
    threshold. The watermark is invisible to a reader but obvious to the score.
    """
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    _grid_ax(ax)
    ax.axvline(4.0, color=MGRAY, ls="--", lw=1.3, zorder=2)
    ax.text(4.0, 1.62, "detection\nthreshold (z=4)", fontsize=9, color=MGRAY,
            ha="center", va="bottom", linespacing=1.1)
    for z, y, col, lbl in [(human_z, 1.0, BLUE, "human"),
                           (watermarked_z, 0.0, RED, "watermarked")]:
        jitter = y + rng.uniform(-0.13, 0.13, len(z))
        ax.scatter(z, jitter, s=120, color=col, alpha=0.85, zorder=3,
                   edgecolor="white", linewidth=1.2)
        ax.text(min(z) - 0.4, y, lbl, fontsize=11.5, fontweight="700",
                color=col, ha="right", va="center")
    ax.set_yticks([])
    ax.set_ylim(-0.6, 1.7)
    ax.set_xlim(min(min(human_z), 0) - 3.0, max(watermarked_z) + 1.2)
    ax.set_xlabel("watermark z-score", fontsize=11.5, color=MGRAY)
    ax.set_title("A Watermark the Eye Can't See, the Math Can", fontsize=14,
                 fontweight="700", color=DGRAY, pad=8)
    plt.tight_layout(pad=0.6)
    _save(fig, save_path)
    plt.show()


# ── Responsible chapter screenings (reuse plot_bakeoff) ───────────────────────

def plot_red_team_zoo(study, save_path="images/responsible/red_team_zoo.png"):
    """Per candidate model: percent of the red-team suite resisted wearing the
    hardened system prompt vs the naive one. The gap up from the faded naive
    bar is what the written security policy buys that model; the height of the
    solid bar is who you would trust with the production job."""
    plot_bakeoff(study, "hardened", "naive",
                 "hardened (security policy)", "naive (no policy)",
                 "Who Holds the Line Under Attack?",
                 "attacks resisted (%)", save_path, full_labels=True)


def plot_poison_zoo(study, save_path="images/responsible/poison_zoo.png"):
    """Per model: how completely it follows the in-context examples it is
    handed, the honest few-shot set vs the label-flipped one. Both bars rise
    together; a model that follows the honest examples to the letter follows the
    poisoned ones just as completely, and a shorter poisoned bar is a model
    leaning on its own prior instead of the flipped labels."""
    plot_bakeoff(study, "clean", "poisoned",
                 "follows honest examples", "follows poisoned examples",
                 "Who Swallows the Poisoned Labels?",
                 "follows the examples (%)", save_path)


def plot_pii_screen(study, save_path="images/responsible/pii_screen.png"):
    """Per model: percent of summaries with zero PII regex hits when politely
    asked to omit identifiers vs when not asked. The faded bar is the default
    behavior (a faithful summarizer parrots the identifiers); the gap up to the
    solid bar is how much a polite request buys, and any solid bar short of 100
    is a model ignoring the request."""
    plot_bakeoff(study, "instructed", "unprompted",
                 "asked to omit identifiers", "not asked",
                 "Can You Just Ask the Model to Redact?",
                 "identifier-free summaries (%)", save_path)


def plot_safety_judges(study, save_path="images/responsible/safety_judges.png"):
    """Per candidate judge: percent of quietly dangerous drafts flagged vs
    percent of safe drafts cleared. A usable judge is tall on both; tall only
    on flags is a paranoid judge that blocks everything, tall only on clears
    is a rubber stamp."""
    plot_bakeoff(study, "flags", "clears",
                 "flags the dangerous", "clears the safe",
                 "Who Can Sit in the Judge's Chair?",
                 "rate (%)", save_path)


# ── Semantics chapter ─────────────────────────────────────────────────────────

def plot_word_vs_embed_similarity(pairs: list, pair_labels: list, save_path=None):
    """Grouped bar chart: word-overlap similarity vs embedding similarity.

    pairs = [(sentence_a, sentence_b), ...]  — similarity is computed internally.
    """
    from genai.embed import similarity as _sim

    words_set  = lambda s: set(s.lower().split())
    word_sims  = [len(words_set(a) & words_set(b)) / len(words_set(a) | words_set(b))
                  for a, b in pairs]
    embed_sims = [_sim(a, b) for a, b in pairs]

    x = np.arange(len(pair_labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(x - w/2, word_sims,  w, label="Word overlap",         color="#e07b54", alpha=0.9)
    ax.bar(x + w/2, embed_sims, w, label="Embedding similarity", color="#4a90d9", alpha=0.9)
    ax.set_xticks(x); ax.set_xticklabels(pair_labels, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Similarity score", fontsize=11)
    ax.set_title("Word Overlap vs. Embedding Similarity", fontsize=13, fontweight="600")
    ax.legend(fontsize=10.5)
    ax.tick_params(labelsize=11)
    ax.axhline(0, color="black", linewidth=0.5)
    plt.tight_layout()
    _save(fig, save_path)
    plt.show()


def plot_crosslingual(study: dict, save_path="images/semantics/crosslingual.png"):
    """Line chart: cosine of an English sentence to its translation in each
    language, traced by two multilingual embedders, nomic-embed-text-v2-moe and
    embeddinggemma. The x-axis runs across languages from the best-carried down
    to Swahili; the unrelated control is drawn as a shaded floor band so each
    language reads against the baseline it would hit by chance.

    Both lines stay high across the well-resourced languages, embeddinggemma above,
    then plunge together at Swahili toward the unrelated floor: meaning transfers
    where the training data was, and runs out at the low-resource edge.
    """
    labels, nomic_v2, gemma = study["labels"], study["nomic_v2"], study["gemma"]
    # The last entry is the unrelated control, drawn as a floor band, not a language.
    langs = labels[:-1]
    nomic_lang, gemma_lang = nomic_v2[:-1], gemma[:-1]
    floor_lo, floor_hi = sorted((nomic_v2[-1], gemma[-1]))
    x = np.arange(len(langs))

    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    _grid_ax(ax)
    # Unrelated-sentence floor: the two controls bracket a thin band; shade the
    # basement below it so anything sinking in is "no better than random."
    ax.axhspan(0, floor_hi, color=MGRAY, alpha=0.13, zorder=0)
    ax.axhline(floor_hi, color=MGRAY, lw=1.1, ls=(0, (5, 3)), zorder=1)
    ax.text(len(langs) - 1.45, floor_hi + 0.015, "unrelated-sentence floor",
            ha="left", va="bottom", fontsize=9, style="italic", color=DGRAY)
    ax.plot(x, gemma_lang, "-o", color=ORANGE, lw=2.4, ms=8, zorder=3, label="embeddinggemma")
    ax.plot(x, nomic_lang, "-o", color=BLUE,   lw=2.4, ms=8, zorder=3,
            label="nomic-embed-text-v2-moe")
    for xi, v in zip(x, gemma_lang):
        ax.text(xi, v + 0.035, f"{v:.2f}", ha="center", va="bottom",
                fontsize=9, fontweight="600", color=ORANGE)
    for xi, v in zip(x, nomic_lang):
        ax.text(xi, v - 0.04, f"{v:.2f}", ha="center", va="top",
                fontsize=9, fontweight="600", color=BLUE)
    ax.set_xticks(x); ax.set_xticklabels(langs, fontsize=10.5, color=DGRAY)
    ax.set_ylabel("cosine to the English sentence", fontsize=11.5, color=MGRAY)
    ax.set_ylim(0, 1.08); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.legend(fontsize=10, framealpha=0, loc="upper right")
    ax.set_title("Does Meaning Survive Translation?",
                 fontsize=12.5, fontweight="600", color=DGRAY, pad=8)
    plt.tight_layout(pad=0.6)
    _save(fig, save_path)
    plt.show()


def plot_toy_embeddings(pts: dict, save_path=None):
    """Scatter of hand-crafted 2D embeddings showing word clustering."""
    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    for word, (x, y) in pts.items():
        ax.scatter(x, y, s=110)
        ax.annotate(word, (x, y), fontsize=11.5,
                    textcoords="offset points", xytext=(5, 4))
    ax.axhline(0, color="#cccccc", linewidth=0.8)
    ax.axvline(0, color="#cccccc", linewidth=0.8)
    ax.tick_params(labelsize=10)
    ax.set(xlim=(-0.3, 1.2), ylim=(-0.55, 1.2))
    ax.set_title("Word Embeddings in 2D", fontsize=13, fontweight="600")
    ax.set_xlabel("Dim 1", fontsize=11); ax.set_ylabel("Dim 2", fontsize=11)
    plt.tight_layout()
    _save(fig, save_path)
    plt.show()


_TSNE_PALETTE = {
    "animals":  "#4a90d9", "vehicles": "#e07b54",
    "fruits":   "#27ae60", "code":     "#8e44ad",
}


def plot_tsne_embeddings(word_groups: dict, palette: dict = None, save_path=None):
    """t-SNE projection of word embeddings, colored by semantic group.

    word_groups = {"group_name": [word, ...], ...}  — embeddings computed internally.
    """
    from sklearn.manifold import TSNE
    from genai.embed import embed as _embed

    if palette is None:
        palette = _TSNE_PALETTE

    words        = [w for ws in word_groups.values() for w in ws]
    group_labels = [g for g, ws in word_groups.items() for _ in ws]
    vecs         = np.array([_embed(w) for w in words])
    coords       = TSNE(n_components=2, random_state=42,
                        perplexity=5).fit_transform(vecs)

    fig, ax = plt.subplots(figsize=(8, 6))
    for grp, color in palette.items():
        mask = [i for i, g in enumerate(group_labels) if g == grp]
        ax.scatter(coords[mask, 0], coords[mask, 1], color=color, s=110, label=grp)
    for i, word in enumerate(words):
        ax.annotate(word, coords[i], fontsize=11,
                    textcoords="offset points", xytext=(4, 4))
    ax.legend(title="Group", framealpha=0.9, fontsize=10.5, title_fontsize=11)
    ax.tick_params(labelsize=10.5)
    ax.set_title("Real Word Embeddings: t-SNE Projection", fontsize=13, fontweight="600")
    ax.set_xlabel("t-SNE dim 1", fontsize=11); ax.set_ylabel("t-SNE dim 2", fontsize=11)
    plt.tight_layout()
    _save(fig, save_path)
    plt.show()


def plot_cosine_angles(save_path=None):
    """Three arrows-from-origin panels illustrating cosine = 1, 0, -1.

    A friendly companion to the formula: two vectors that point the same way
    score 1, that sit at a right angle score 0, and that point in opposite
    directions score -1. The first panel deliberately draws two arrows of
    different lengths to show that only direction matters, not magnitude.
    """
    A, B = BLUE, ORANGE
    panels = [
        ("Same direction", "+1", [(0.92, 0.66, A), (0.58, 0.42, B)]),
        ("Right angle",     "0", [(0.95, 0.00, A), (0.00, 0.95, B)]),
        ("Opposite",       "-1", [(0.80, 0.60, A), (-0.80, -0.60, B)]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.3))
    for ax, (caption, cos, vecs) in zip(axes, panels):
        for x, y, col in vecs:
            ax.annotate("", xy=(x, y), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=2.6,
                                        mutation_scale=20, shrinkA=0, shrinkB=0))
        ax.scatter([0], [0], s=16, color=DGRAY, zorder=5)
        ax.set(xlim=(-1.2, 1.2), ylim=(-1.2, 1.2))
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(caption, fontsize=12.5, fontweight="700", color=DGRAY, pad=4)
        ax.text(0, -1.08, f"cosine = {cos}", ha="center", va="center",
                fontsize=12, fontweight="600", color=MGRAY)
    fig.suptitle("Cosine similarity is the angle between two vectors",
                 fontsize=13, fontweight="600", color=DGRAY, y=1.02)
    plt.tight_layout()
    _save(fig, save_path)
    plt.show()


def plot_analogy_offsets(save_path=None):
    """Why an analogy carries from one pair to another, drawn as arrows.

    Each panel measures the step ``a -> b`` for two pairs and draws them from a
    common origin at their TRUE angle apart (arccos of their cosine). When the
    arrows point the same way the analogy transfers; gender keeps a tight cone,
    opposites splay toward a right angle, so 'the opposite of' is a different
    arrow for every word. Angles are read from real embeddings, not posed.
    """
    from genai.embed import embed, similarity
    step = lambda a, b: embed(b) - embed(a)

    # Each arrow: word_a, word_b, color, (label_x, label_y, ha, va).
    # The first arrow in each panel is the reference, drawn along +x.
    panels = [
        ("Gender", [
            ("man", "woman", BLUE,     (0.55, -0.17, "center", "top")),
            ("actor", "actress", GREEN, (0.74, 0.73, "left",   "center")),
            ("king", "queen", ORANGE,  (0.40, 0.99, "center", "bottom")),
        ]),
        ("Comparative", [
            ("big", "bigger", BLUE,      (0.55, -0.17, "center", "top")),
            ("tall", "taller", GREEN,    (0.70, 0.80, "left",   "center")),
            ("small", "smaller", ORANGE, (0.18, 1.04, "center", "bottom")),
        ]),
        ("Opposite", [
            ("happy", "sad", BLUE,   (0.55, -0.17, "center", "top")),
            ("rich", "poor", ORANGE, (0.38,  1.00, "left",  "bottom")),
            ("hot", "cold", GREEN,   (-0.05, 1.15, "right", "bottom")),
        ]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.9))
    for ax, (title, arrows) in zip(axes, panels):
        ref = step(arrows[0][0], arrows[0][1])
        drawn = []
        for i, (a, b, col, lab) in enumerate(arrows):
            cos = 1.0 if i == 0 else similarity(ref, step(a, b))
            drawn.append((np.degrees(np.arccos(np.clip(cos, -1, 1))), a, b, col, lab))

        spread = max(ang for ang, *_ in drawn)
        ax.add_patch(mp.Wedge((0, 0), 1.0, 0, spread, color=LGRAY, zorder=0))
        ax.plot([0, 0], [0, 1.18], ls=(0, (2, 3)), color=MGRAY, lw=1, zorder=1)
        ax.text(0.03, 1.2, "90°", fontsize=8.5, color=MGRAY, ha="left", va="bottom")

        for ang, a, b, col, (lx, ly, ha, va) in drawn:
            r = np.radians(ang)
            ax.annotate("", xy=(np.cos(r), np.sin(r)), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=2.4,
                                        mutation_scale=15, shrinkA=0, shrinkB=0))
            ax.text(lx, ly, f"{a}→{b}", fontsize=9.5, fontweight="700",
                    color=col, ha=ha, va=va)
        ax.scatter([0], [0], s=14, color=DGRAY, zorder=5)
        ax.set(xlim=(-0.5, 1.5), ylim=(-0.34, 1.32))
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, fontsize=12.5, fontweight="700", color=DGRAY, pad=6)

    fig.suptitle("An analogy carries only when both pairs move along the same arrow",
                 fontsize=12.5, fontweight="600", color=DGRAY, y=1.0)
    plt.tight_layout()
    _save(fig, save_path)
    plt.show()


def plot_gap_comparison(save_path=None):
    """Dumbbell chart of the related-vs-unrelated similarity gap, two ways.

    For each pair, a line runs from the unrelated score (orange) to the related
    score (blue); its length is the gap. Averaging crams every pair into a short
    span high on the scale, while the transformer pulls related and unrelated
    apart into long spans, so its lines are visibly longer. Real scores from
    genai.embed.GAP_PAIRS, sorted by the transformer gap.
    """
    from genai import similarity
    from genai.embed import averaged_similarity, GAP_PAIRS
    rows = [(lab, averaged_similarity(a, rel), averaged_similarity(a, unr),
             similarity(a, rel), similarity(a, unr)) for lab, a, rel, unr in GAP_PAIRS]
    rows.sort(key=lambda r: r[3] - r[4])
    labels = [r[0] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8), sharey=True)
    panels = [("Averaging word vectors", [(r[2], r[1]) for r in rows]),
              ("Transformer (nomic-embed-text)", [(r[4], r[3]) for r in rows])]
    for ax, (title, spans) in zip(axes, panels):
        for i, (unr, rel) in enumerate(spans):
            ax.plot([unr, rel], [i, i], color=LGRAY, lw=3.5, zorder=1,
                    solid_capstyle="round")
            ax.scatter([unr], [i], color=ORANGE, s=60, zorder=3)
            ax.scatter([rel], [i], color=BLUE, s=60, zorder=3)
        mean_gap = np.mean([rel - unr for unr, rel in spans])
        ax.set_title(f"{title}\nmean gap {mean_gap:.2f}",
                     fontsize=11.5, fontweight="700", color=DGRAY)
        ax.set_xlim(0, 1)
        ax.set_xlabel("cosine similarity", fontsize=10.5)
        ax.xaxis.grid(True, color=LGRAY, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(left=False)
    axes[0].set_yticks(range(len(labels)))
    axes[0].set_yticklabels(labels, fontsize=10.5)
    axes[0].set_ylim(-0.6, len(labels) - 0.4)

    from matplotlib.lines import Line2D
    dot = lambda c, l: Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                              markersize=9, label=l)
    axes[0].legend(handles=[dot(ORANGE, "unrelated"), dot(BLUE, "related")],
                   loc="lower left", bbox_to_anchor=(-0.5, 1.0), ncol=2,
                   fontsize=10, frameon=False, columnspacing=1.3, handletextpad=0.4)
    fig.suptitle("The transformer opens a wider gap between related and unrelated text",
                 fontsize=12.5, fontweight="600", color=DGRAY, y=1.02)
    plt.tight_layout()
    _save(fig, save_path)
    plt.show()


def plot_next_token(prompt: str, dist: list, save_path=None):
    """Horizontal bar chart of a next-token probability distribution.

    dist = [(token, probability), ...] from genai.next_token_distribution,
    most likely first. The top candidate is highlighted; the rest fade gray.
    """
    labels = [t if t.strip() else repr(t) for t, _ in dist][::-1]
    probs  = [p * 100 for _, p in dist][::-1]
    colors = [MGRAY] * (len(probs) - 1) + [BLUE]   # winner sits at the top

    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.barh(range(len(probs)), probs, color=colors, zorder=3)
    for i, p in enumerate(probs):
        ax.text(p + 1.5, i, f"{p:.1f}%", va="center", fontsize=11, color=DGRAY)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=12)
    ax.set_xlim(0, 100)
    ax.set_xlabel("probability of being the next token (%)", fontsize=11)
    ax.set_title(f'"{prompt} ___"', fontsize=13, color=DGRAY, loc="left", pad=10)
    _grid_ax(ax)
    ax.xaxis.grid(True, color=LGRAY, zorder=0)
    ax.yaxis.grid(False)
    plt.tight_layout()
    _save(fig, save_path)
    plt.show()


def plot_two_step(prompt, top_word, step1, step2, save_path=None):
    """Two next-token distributions stacked, showing generation one step at a time.

    Top panel: the model's distribution over the FIRST word after `prompt`, an
    open, spread-out field. Bottom panel: once we commit to its top pick,
    `top_word`, the model forecasts the SECOND word, and the field can snap
    sharply onto a single front-runner. The committed word is the blue bar in
    the top panel and the carried-over word in the bottom panel's title.

    step1, step2 = [(token, probability), ...] from genai.next_token_distribution.
    """
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(7, 6.8))
    panels = [
        (ax_top, step1, f'Step 1   "{prompt} ___"', "probability of the first word (%)"),
        (ax_bot, step2, f'Step 2   "{prompt} {top_word} ___"', "probability of the second word (%)"),
    ]
    for ax, dist, title, xlabel in panels:
        labels = [t if t.strip() else repr(t) for t, _ in dist][::-1]
        probs  = [p * 100 for _, p in dist][::-1]
        colors = [MGRAY] * (len(probs) - 1) + [BLUE]   # winner sits at the top
        ax.barh(range(len(probs)), probs, color=colors, zorder=3)
        for i, p in enumerate(probs):
            ax.text(p + 1.5, i, f"{p:.0f}%", va="center", fontsize=11, color=DGRAY)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=12)
        ax.set_xlim(0, 100)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_title(title, fontsize=12.5, color=DGRAY, loc="left", pad=10)
        _grid_ax(ax)
        ax.xaxis.grid(True, color=LGRAY, zorder=0)
        ax.yaxis.grid(False)
    plt.tight_layout(h_pad=2.2)
    _save(fig, save_path)
    plt.show()


# ── Prompting chapter ─────────────────────────────────────────────────────────

def plot_temperature_lifecycle(save_path="images/prompting/temperature_lifecycle.png"):
    """Recommended sampling temperature across the software development lifecycle.

    A high temperature suits open-ended phases (brainstorming, design); a low one
    suits phases that demand consistency (implementation, debugging). Test-case
    generation sits in the middle — you want some variety in the cases.
    """
    phases = ["Brainstorm", "Design", "Implement", "Generate\nTests", "Debug"]
    temps  = [0.9, 0.7, 0.2, 0.6, 0.1]
    x = np.arange(len(phases))

    fig, ax = plt.subplots(figsize=(9, 4.4))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")

    # Shade the "explore" (high) and "be consistent" (low) bands.
    ax.axhspan(0.55, 1.08, color=ORANGE, alpha=0.07, zorder=0)
    ax.axhspan(-0.05, 0.40, color=BLUE,  alpha=0.07, zorder=0)
    ax.text(len(phases) - 0.4, 1.0, "explore", color=ORANGE,
            fontsize=12, style="italic", ha="right", va="top")
    ax.text(len(phases) - 0.4, 0.02, "be consistent", color=BLUE,
            fontsize=12, style="italic", ha="right", va="bottom")

    pt_cols = [ORANGE if t >= 0.5 else BLUE for t in temps]
    ax.plot(x, temps, color=MGRAY, lw=2, zorder=2)
    ax.scatter(x, temps, c=pt_cols, s=130, zorder=3,
               edgecolors="white", linewidths=1.2)
    for xi, t in zip(x, temps):
        ax.text(xi, t + 0.07, f"{t:.1f}", ha="center", fontsize=12,
                fontweight="600", color=DGRAY, zorder=4)

    ax.set_xticks(x); ax.set_xticklabels(phases, fontsize=12, color=DGRAY)
    ax.set_ylim(-0.05, 1.14)
    ax.set_ylabel("recommended temperature", fontsize=12, color=MGRAY)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_color(LGRAY); ax.spines["bottom"].set_color(LGRAY)
    ax.tick_params(colors=MGRAY, labelsize=11)
    ax.set_title("Temperature Across the Development Lifecycle",
                 fontsize=14, fontweight="600", color=DGRAY, pad=10)
    plt.tight_layout(pad=0.5)
    _save(fig, save_path)
    plt.show()


def plot_sampling_controls(dist, k=4, p=0.9, n=7, temps=(0.5, 1.0, 1.5),
                           save_path="images/prompting/sampling_controls.png"):
    """The three control-panel dials acting on one real next-token distribution.

    `dist` is a list of (token, probability) pairs from
    genai.next_token_distribution -- the model's own ranked guesses for the word
    that comes next. We keep the top `n`, renormalize, and show the same numbers
    two ways, mirroring the chapter's split between weighing and counting.

    Left: temperature reshapes the odds. A cool setting (T < 1) sharpens the
    distribution toward the front-runner; a hot one (T > 1) flattens it so the
    long-shots get a real say. Same candidates, different boldness.

    Right: top-k and top-p decide which candidates are even on the table. Top-k
    keeps a fixed count (the k tallest); top-p keeps the smallest group whose
    probabilities clear p, so it widens when the model is unsure. They are set
    here to disagree on purpose, which is the whole point of the panel.
    """
    toks = [t if t.strip() else repr(t) for t, _ in dist[:n]]
    base = np.array([pr for _, pr in dist[:n]], dtype=float)
    base = base / base.sum()                       # renormalize over shown candidates
    x = np.arange(n)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10, 4.2))
    fig.patch.set_facecolor("white")

    # ── Left: one distribution, reshaped by temperature ──────────────────────
    logits = np.log(base)
    for T in temps:
        z = np.exp((logits - logits.max()) / T)
        probs = z / z.sum()
        col = BLUE if T < 1 else ORANGE if T > 1 else DGRAY
        axL.plot(x, probs * 100, color=col, lw=2.6 if T != 1 else 1.8,
                 marker="o", ms=6, zorder=3, label=f"T = {T:.1f}")
    _grid_ax(axL)
    axL.set_xticks(x); axL.set_xticklabels(toks, rotation=30, ha="right", fontsize=10)
    axL.set_ylabel("probability (%)", fontsize=11, color=MGRAY)
    axL.set_title("Temperature reshapes the odds", fontsize=12.5,
                  fontweight="600", color=DGRAY, pad=8)
    axL.legend(fontsize=10.5, frameon=False, loc="upper right")

    # ── Right: the same distribution, trimmed by top-k and top-p ─────────────
    m = int(np.searchsorted(np.cumsum(base), p) + 1)   # smallest nucleus clearing p
    axR.bar(x, base * 100, color=[BLUE if i < m else MGRAY for i in range(n)], zorder=3)
    axR.axvline(k - 0.5, color=ORANGE, lw=2.2, ls="--", zorder=4)
    _grid_ax(axR)
    top = base.max() * 100
    axR.set_ylim(0, top * 1.24)
    # Keep both labels off the dashed line at x = k - 0.5: top-k to its right,
    # top-p in the upper-left over the blue nucleus (a vertical line crosses any
    # text sitting at its x, whatever the height).
    axR.text(k - 0.5 + 0.12, top * 1.15, f"top-k = {k}", color=ORANGE, fontsize=11,
             fontweight="700", ha="left")
    axR.text(-0.4, top * 1.15, f"top-p = {p:g}  keeps {m}", color=BLUE,
             fontsize=11, fontweight="700", ha="left")
    axR.set_xticks(x); axR.set_xticklabels(toks, rotation=30, ha="right", fontsize=10)
    axR.set_ylabel("probability (%)", fontsize=11, color=MGRAY)
    axR.set_title("Top-k and top-p trim the field", fontsize=12.5,
                  fontweight="600", color=DGRAY, pad=8)

    plt.tight_layout(pad=1.0)
    _save(fig, save_path)
    plt.show()


def plot_token_boundaries(text=None, lines=None, tokenizer="general",
                          title="Same Code, Two Tokenizers",
                          save_path="images/tokens/token_boundaries.png"):
    """Each token as its own colored chip, so "tokens are not words" is visible.

    Renders one or more strings as a left-to-right strip of chips, one chip per
    token, with chip width tracking the token's length. A leading space inside a
    token is drawn as a faint dot, so the word-boundary tokens that tiktoken
    emits (like " return") stay legible instead of looking clipped.

    Pass ``text`` to run one line through both the general-purpose and the code
    tokenizer, the side-by-side comparison the chapter leans on. For full control,
    pass ``lines`` instead: each row's value is either a string (tokenized with the
    function-level ``tokenizer``) or a ``(text, tokenizer)`` pair.
    """
    from genai.tokens import tokenize
    if lines is None:
        if text is None:
            text = "def get_user_by_id(user_id):"
        lines = {
            "general": (text, "general"),
            "code":    (text, "code"),
        }
    # Normalize each row to (label, text, tokenizer).
    rows = [(label, *(v if isinstance(v, tuple) else (v, tokenizer)))
            for label, v in lines.items()]
    palette = [BLUE, GREEN, PURPLE, ORANGE]
    cw, pad, gap = 0.34, 0.22, 0.12          # char width, chip padding, inter-chip gap
    show = lambda t: t.replace(" ", "·").replace("\n", "⏎") or "·"

    end = []
    for _, text, tok in rows:
        x = 0.0
        for piece in tokenize(text, tok):
            x += len(show(piece)) * cw + 2 * pad + gap
        end.append(x)
    span = max(end)

    fig, ax = plt.subplots(figsize=(9.6, 0.62 + 0.92 * len(rows)))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.set_xlim(-2.0, span + 2.7); ax.axis("off")
    ax.set_ylim(0, len(rows))

    for r, (label, text, tok) in enumerate(rows):
        y = len(rows) - r - 0.5
        ax.text(-1.9, y, label, ha="left", va="center", fontsize=11.5,
                fontweight="700", color=DGRAY, family="monospace")
        x = 0.0
        for i, piece in enumerate(tokenize(text, tok)):
            disp = show(piece)
            w = len(disp) * cw + 2 * pad
            _fbox(ax, x + w / 2, y, w / 2, 0.30, palette[i % len(palette)])
            ax.text(x + w / 2, y, disp, ha="center", va="center", fontsize=11.5,
                    color="white", family="monospace", zorder=4)
            x += w + gap
        ax.text(x + 0.15, y, f"{i + 1} tokens", ha="left", va="center",
                fontsize=9.5, color=MGRAY, style="italic")

    ax.set_title(title, fontsize=13, fontweight="700",
                 color=DGRAY, pad=8)
    plt.tight_layout(pad=0.5)
    _save(fig, save_path)
    plt.show()


def plot_special_tokens(tokenizer="code",
                        save_path="images/tokens/special_tokens.png"):
    """Every reserved control token, grouped by the repository part it marks.

    Sorting the vocabulary's special tokens into families turns a flat list into
    a map: the code tokenizer carries dedicated markers for issue threads, pull
    requests, Jupyter notebooks, fill-in-the-middle, and even redacted secrets.
    Rules run in order, first match wins, and anything unmatched lands in "Other"
    so the figure stays honest to whatever the tokenizer actually ships.
    """
    from genai.tokens import special_tokens
    toks = special_tokens(tokenizer)
    groups = [
        ("Document boundaries", BLUE,   lambda t: t in ("<|endoftext|>", "<empty_output>")),
        ("Fill-in-the-middle",  GREEN,  lambda t: t.startswith("<fim_")),
        ("Repository layout",   ORANGE, lambda t: t in ("<repo_name>", "<file_sep>")),
        ("Issue threads",       PURPLE, lambda t: t.startswith("<issue_")),
        ("Jupyter notebooks",   TEAL,   lambda t: t.startswith("<jupyter")),
        ("Code transforms",     AMBER,  lambda t: "intermediate" in t),
        ("Pull requests",       PINK,   lambda t: t.startswith("<pr")),
        ("Redacted secrets",    RED,    lambda t: t in ("<NAME>", "<EMAIL>", "<KEY>", "<PASSWORD>")),
    ]
    buckets = {name: [] for name, _, _ in groups}
    order = list(groups)
    for t in toks:
        for name, _, rule in groups:
            if rule(t):
                buckets[name].append(t); break
        else:
            buckets.setdefault("Other", []).append(t)
    if buckets.get("Other"):
        order.append(("Other", MGRAY, None))

    # Flow layout: heading, then chips wrapping at a fixed content width.
    cw, pad, gap = 0.70, 0.55, 0.55           # char width, chip padding, inter-chip gap
    line_h, head_h, cat_gap = 1.35, 1.45, 0.6
    W = 60.0
    chips, heads = [], []                     # (cx, cy, hw, text, color) / (y, text, color, n)
    y = 0.0
    for name, color, _ in order:
        items = buckets[name]
        if not items:
            continue
        heads.append((y, f"{name}  ({len(items)})", color))
        y -= head_h
        x = 0.0
        for t in items:
            w = len(t) * cw + 2 * pad
            if x > 0 and x + w > W:
                x = 0.0; y -= line_h
            chips.append((x + w / 2, y, w / 2, t, color))
            x += w + gap
        y -= line_h + cat_gap
    total_h = -y + 0.4

    fig_w = 9.2
    fig, ax = plt.subplots(figsize=(fig_w, total_h * fig_w / (W + 6)))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.set_xlim(-1.0, W + 5.0); ax.set_ylim(-total_h + 0.2, head_h + 0.4)
    ax.axis("off")
    for y0, text, color in heads:
        ax.text(0.0, y0, text, ha="left", va="center", fontsize=11.5,
                fontweight="700", color=color)
    for cx, cy, hw, text, color in chips:
        _fbox(ax, cx, cy, hw, 0.44, color)
        ax.text(cx, cy, text, ha="center", va="center", fontsize=11,
                fontweight="bold", color="white", family="monospace", zorder=4)
    ax.set_title(f"What {len(toks)} Special Tokens Reveal",
                 fontsize=13, fontweight="700", color=DGRAY, pad=8)
    plt.tight_layout(pad=0.5)
    _save(fig, save_path)
    plt.show()


def plot_prompt_anatomy(save_path="images/prompting/prompt_anatomy.png"):
    """One good prompt, broken into the four parts you can tighten one at a time.

    Instruction (what to do), context (what the model cannot read from your mind),
    output format (the shape you want back), and a few examples. The worked example
    is a small support-ticket urgency classifier. Conceptual diagram, no model call.
    """
    parts = [
        ("INSTRUCTION",   "Sort each support ticket by urgency.",              BLUE,   "#DBEAFE"),
        ("CONTEXT",       "Small team; P1 means a customer is fully blocked.", GREEN,  "#DCFCE7"),
        ("OUTPUT FORMAT", "Reply with one word: high, medium, or low.",        ORANGE, "#FFEDD5"),
        ("EXAMPLES",      '"The checkout page is down"  →  high',              PURPLE, "#EDE9FE"),
    ]
    fig, ax = plt.subplots(figsize=(9.4, 4.0))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 14); ax.set_ylim(0, 4.3); ax.axis("off")
    ax.text(7, 4.06, "The Anatomy of a Prompt", ha="center",
            fontsize=14, fontweight="700", color=DGRAY)
    for (name, text, col, bg), y in zip(parts, [3.25, 2.45, 1.65, 0.85]):
        ax.text(0.2, y, name, ha="left", va="center",
                fontsize=11.5, fontweight="700", color=col)
        _fbox(ax, 8.7, y, 4.9, 0.33, bg, ec=col, lw=1.4)
        ax.text(4.0, y, text, ha="left", va="center", fontsize=11, color=DGRAY)
    plt.tight_layout(pad=0.6)
    _save(fig, save_path)
    plt.show()


def plot_token_pipeline(word="strawberry",
                        save_path="images/tokens/token_pipeline.png"):
    """Text -> subword tokens -> integer IDs -> model. The model only sees numbers."""
    from genai.tokens import tokenize, token_ids
    pieces, ids = tokenize(word), token_ids(word)

    fig, ax = plt.subplots(figsize=(9.5, 3.9))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.set_xlim(0, 14); ax.set_ylim(0, 4); ax.axis("off")

    y, xs = 2.35, [1.9, 5.3, 8.7, 12.1]
    stages = [
        (xs[0], f'"{word}"',            LGRAY,  DGRAY,   "raw text"),
        (xs[1], "  ".join(pieces),      BLUE,   "white", "subword tokens"),
        (xs[2], str(ids),               PURPLE, "white", "integer IDs"),
        (xs[3], "Language\nModel",      GREEN,  "white", "sees only numbers"),
    ]
    for cx, lbl, fc, tc, sub in stages:
        _fbox(ax, cx, y, 1.30, 0.58, fc)
        _label(ax, cx, y, lbl, tc=tc, fs=11)
        ax.text(cx, y - 1.02, sub, ha="center", fontsize=9.5,
                color=MGRAY, style="italic")
    for i in range(len(xs) - 1):
        _arr(ax, xs[i], y, xs[i + 1], y, pct=0.41)

    ax.set_title("From Text to Tokens to Numbers",
                 fontsize=13, fontweight="700", color=DGRAY, pad=6)
    plt.tight_layout(pad=0.5)
    _save(fig, save_path)
    plt.show()


def plot_multilingual_tokens(save_path="images/tokens/multilingual_tokens.png"):
    """Grouped bars: the same sentence costs more tokens in some languages."""
    from genai.tokens import count_tokens
    sentences = {
        "English": "The quick brown fox jumps over the lazy dog.",
        "Spanish": "El veloz zorro marrón salta sobre el perro perezoso.",
        "Hindi":   "तेज़ भूरी लोमड़ी आलसी कुत्ते के ऊपर कूदती है।",
    }
    langs   = list(sentences)
    general = [count_tokens(s) for s in sentences.values()]
    multi   = [count_tokens(s, "multilingual") for s in sentences.values()]

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    fig.patch.set_facecolor("white"); _grid_ax(ax)
    x, w = np.arange(len(langs)), 0.38
    bars = [ax.bar(x - w/2, general, w, color=MGRAY, label="general (GPT-4)", zorder=3),
            ax.bar(x + w/2, multi,   w, color=BLUE,  label="multilingual (GPT-4o)", zorder=3)]
    top = max(general)
    for group in bars:
        for r in group:
            ax.text(r.get_x() + r.get_width()/2, r.get_height() + top*0.02,
                    f"{int(r.get_height())}", ha="center",
                    fontsize=10.5, fontweight="600", color=DGRAY)
    ax.set_xticks(x); ax.set_xticklabels(langs, fontsize=12, color=DGRAY)
    ax.set_ylabel("tokens for the same sentence", fontsize=11, color=MGRAY)
    ax.set_ylim(0, top * 1.18)
    ax.legend(frameon=False, fontsize=10)
    ax.set_title("The Same Sentence Costs More in Some Languages",
                 fontsize=13, fontweight="700", color=DGRAY, pad=8)
    plt.tight_layout(pad=0.6)
    _save(fig, save_path)
    plt.show()


def plot_code_scorecard(grid, save_path="images/metacoding/code_scorecard.png"):
    """Heatmap of per-category scores: model (rows) against task column, by mode.

    grid = [(model_label, [(solved, total), ...]), ...] in CODE_COLUMNS order, each
    cell the count of small tasks in that category the model fully solved. Cells run
    red (none) through orange and yellow to green (all); a category a model has no
    mode for (fill-in-the-middle on an instruction-only model) is gray and dashed.
    The point is the texture: different models light up different columns, so
    capability reads as a profile, not a single verdict.
    """
    from genai.code import CODE_COLUMNS
    labels = [g[0] for g in grid]
    cols   = [c for c, _m, _k in CODE_COLUMNS]
    modes  = [m for _c, m, _k in CODE_COLUMNS]
    n_rows, n_cols = len(labels), len(cols)

    frac  = np.full((n_rows, n_cols), np.nan)
    annot = [["" for _ in range(n_cols)] for _ in range(n_rows)]
    for i, (_, row) in enumerate(grid):
        for j, (solved, total) in enumerate(row):
            if solved is None:
                annot[i][j] = "—"
            else:
                frac[i, j]  = solved / total
                annot[i][j] = f"{solved}/{total}"

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad(LGRAY)
    ax.imshow(np.ma.masked_invalid(frac), cmap=cmap, vmin=0, vmax=1, aspect="auto")

    for i in range(n_rows):
        for j in range(n_cols):
            val = frac[i, j]
            tc = "white" if (val == val and val < 0.2) else DGRAY
            ax.text(j, i, annot[i][j], ha="center", va="center",
                    fontsize=11, color=tc, fontweight="600")

    ax.set_xticks(range(n_cols)); ax.set_xticklabels(cols, fontsize=10.5)
    ax.set_yticks(range(n_rows)); ax.set_yticklabels(labels, fontsize=10.5)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)

    n_instr = modes.count("instruction")
    ax.axvline(n_instr - 0.5, color="white", lw=5)
    ax.text((n_instr - 1) / 2, -0.82, "Instruction (write the function)",
            ha="center", fontsize=10.5, fontweight="700", color=DGRAY)
    ax.text(n_instr + (n_cols - n_instr - 1) / 2, -0.82, "Fill-in-the-Middle",
            ha="center", fontsize=10.5, fontweight="700", color=DGRAY)
    ax.set_ylim(n_rows - 0.5, -1.35)

    ax.set_title("Code zoo scorecard: small tasks solved per category, by model",
                 fontsize=12.5, fontweight="700", color=DGRAY, pad=24)
    plt.tight_layout()
    _save(fig, save_path)
    plt.show()


def plot_three_pass(save_path="images/research/three_pass.png"):
    """The three-pass reading method as a narrowing funnel.

    Each pass costs more time and admits fewer papers: every paper gets a
    10-minute first pass, the survivors earn a 30-minute second pass, and only
    the few you build on get the deep third pass. Boxes shrink left to right to
    make the funnel felt rather than stated.
    """
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.set_xlim(0, 14.5); ax.set_ylim(0, 5); ax.axis("off")

    y = 2.6
    passes = [
        (4.0,  1.45, 1.02, BLUE,   "First Pass\n10 min",
         "title · abstract\nintro · conclusion"),
        (8.0,  1.40, 0.80, ORANGE, "Second Pass\n30 min",
         "figures · tables\nrelated work"),
        (12.0, 1.35, 0.56, GREEN,  "Third Pass\n1-3 hrs",
         "re-implement\nthe key ideas"),
    ]
    y_sub = 1.15
    for cx, hw, hh, fc, lbl, sub in passes:
        _fbox(ax, cx, y, hw, hh, fc)
        _label(ax, cx, y, lbl, tc="white", fs=11.5)
        ax.text(cx, y_sub, sub, ha="center", va="top", fontsize=9.0,
                color=MGRAY, style="italic", linespacing=1.25)

    # "every paper" feeds the first pass
    ax.text(1.0, y, "every\npaper", ha="center", va="center", fontsize=9.0,
            color=MGRAY, style="italic", linespacing=1.2)
    _arr(ax, 1.65, y, 2.55, y, pct=0.10)

    # narrowing arrows between passes, captioned with what survives each cut
    for x1, x2, cx, cap in [(5.45, 6.60, 6.02, "worth a\ncloser look"),
                            (9.40, 10.65, 10.02, "worth a\ndeep read")]:
        _arr(ax, x1, y, x2, y, pct=0.12)
        ax.text(cx, 4.10, cap, ha="center", va="center", fontsize=8.5,
                color=DGRAY, style="italic", linespacing=1.15)

    ax.set_title("The Three-Pass Method: Each Pass Costs More, Admits Fewer",
                 fontsize=12.5, fontweight="700", color=DGRAY, pad=10)
    plt.tight_layout(pad=0.5)
    _save(fig, save_path)
    plt.show()


# ── Thinking chapter ──────────────────────────────────────────────────────────

def plot_thinking_latency(rows: list, save_path="images/thinking/latency.png"):
    """Grouped bars: time to answer for a standard vs a thinking model, per task.

    rows = [{"label": str, "std_s": float, "thk_s": float}, ...]
    The gap widens with difficulty: thinking is nearly free on a fact lookup and
    expensive on a multi-step puzzle, because the reasoning chain grows with it.
    """
    labels = [r["label"] for r in rows]
    std    = [r["std_s"] for r in rows]
    thk    = [r["thk_s"] for r in rows]
    x = np.arange(len(rows)); w = 0.36

    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    _grid_ax(ax)
    bars = [ax.bar(x - w/2, std, w, color=BLUE,   zorder=3, label="standard model"),
            ax.bar(x + w/2, thk, w, color=ORANGE, zorder=3, label="thinking model")]
    top = max(std + thk)
    for group in bars:
        for bar in group:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + top*0.02,
                    f"{bar.get_height():.1f}s", ha="center", va="bottom",
                    fontsize=10, fontweight="600", color=DGRAY)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10.5, color=DGRAY)
    ax.set_ylabel("time to answer (s)", fontsize=11.5, color=MGRAY)
    ax.set_ylim(0, top * 1.20)
    ax.legend(fontsize=10.5, framealpha=0, loc="upper left")
    ax.set_title("Thinking Is Far Slower, Easy Question or Hard",
                 fontsize=12.5, fontweight="600", color=DGRAY, pad=8)
    plt.tight_layout(pad=0.6)
    _save(fig, save_path)
    plt.show()


def plot_thinking_accuracy(labels: list, std_acc: list, thk_acc: list,
                           runs: int, save_path="images/thinking/accuracy.png",
                           series=("snap answer", "thinking"),
                           title="Novel Multi-Step Problems: Snap Answers vs Thinking"):
    """Grouped bars: accuracy (% correct over `runs` trials) on novel problems.

    std_acc / thk_acc are fractions in [0, 1], one per problem in `labels`.
    The standard model answers in a single snap; the thinking model reasons
    internally first. The gap shows up only where a snap answer is unsafe.
    """
    std = [100 * a for a in std_acc]
    thk = [100 * a for a in thk_acc]
    x = np.arange(len(labels)); w = 0.36

    fig, ax = plt.subplots(figsize=(7.8, 4.0))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    _grid_ax(ax)
    bars = [ax.bar(x - w/2, std, w, color=BLUE,   zorder=3, label=series[0]),
            ax.bar(x + w/2, thk, w, color=ORANGE, zorder=3, label=series[1])]
    for group in bars:
        for bar in group:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f"{bar.get_height():.0f}", ha="center", va="bottom",
                    fontsize=9.5, fontweight="600", color=DGRAY)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10, color=DGRAY)
    ax.set_ylabel(f"accuracy over {runs} runs (%)", fontsize=11.5, color=MGRAY)
    ax.set_ylim(0, 132); ax.set_yticks([0, 25, 50, 75, 100])
    ax.legend(fontsize=10.5, framealpha=0, loc="upper center", ncol=2)
    ax.set_title(title,
                 fontsize=12.5, fontweight="600", color=DGRAY, pad=8)
    plt.tight_layout(pad=0.6)
    _save(fig, save_path)
    plt.show()


def plot_self_consistency(study: dict,
                          save_path="images/thinking/self_consistency.png"):
    """Accuracy as a majority vote widens from 1 to N sampled reasoning chains.

    ``study`` carries n_values and a {problem: [accuracy per n]} curve map. A
    vote climbs where the model wobbles toward the right answer (the curve starts
    above the noise and rises), is redundant where the model is already sure (flat
    along the top), and powerless where it has no real signal (flat near the floor).
    """
    from matplotlib.ticker import PercentFormatter
    n = study["n_values"]
    colors = {"Trains": BLUE, "Ages": GREEN, "Handshakes": PURPLE, "Well": RED}
    cycle = [GREEN, PURPLE, BLUE, RED, ORANGE]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    _grid_ax(ax)
    for i, (name, curve) in enumerate(study["curves"].items()):
        col = colors.get(name, cycle[i % len(cycle)])
        ax.plot(n, curve, color=col, marker="o", lw=2.2, ms=7, zorder=3)
        ax.text(n[-1] + 0.15, curve[-1], f" {name}", color=col,
                va="center", ha="left", fontsize=11, fontweight="600")

    ax.set_xlabel("reasoning chains sampled, then voted", fontsize=11, color=MGRAY)
    ax.set_ylabel("accuracy", fontsize=11, color=MGRAY)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0.6, n[-1] + 1.9)
    ax.set_xticks(n)
    ax.set_title("Thinking Wider: Voting Helps Only Where the Model Wobbles",
                 fontsize=13.5, fontweight="600", color=DGRAY, pad=8)
    plt.tight_layout(pad=0.4)
    _save(fig, save_path)
    plt.show()


def plot_step_back(study: dict, save_path="images/prompting/step_back.png"):
    """Grouped bars: accuracy of a cold answer vs a step-back answer per problem.

    ``study`` carries labels and two accuracy lists (fractions in [0, 1]).
    Step-back climbs where the model knows the principle but its snap answer
    follows the wrong intuition (Half, Pendulum), helps only partway where the
    arithmetic still trips it (Pressure), and is redundant where the cold answer
    is already right (Inverse).
    """
    labels = study["labels"]
    direct = [100 * a for a in study["direct"]]
    step = [100 * a for a in study["step_back"]]
    x = np.arange(len(labels)); w = 0.36

    fig, ax = plt.subplots(figsize=(8, 4.2))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    _grid_ax(ax)
    bars = [ax.bar(x - w/2, direct, w, color=BLUE,   zorder=3, label="cold answer"),
            ax.bar(x + w/2, step,   w, color=ORANGE, zorder=3, label="step-back")]
    for group in bars:
        for bar in group:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f"{bar.get_height():.0f}", ha="center", va="bottom",
                    fontsize=9.5, fontweight="600", color=DGRAY)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10, color=DGRAY)
    ax.set_ylabel(f"accuracy over {study['samples']} tries (%)",
                  fontsize=11.5, color=MGRAY)
    ax.set_ylim(0, 118); ax.set_yticks([0, 25, 50, 75, 100])
    ax.legend(fontsize=10.5, framealpha=0, loc="upper center", ncol=2)
    ax.set_title("Step-Back Prompting: Naming the Principle Before Answering",
                 fontsize=12.5, fontweight="600", color=DGRAY, pad=8)
    plt.tight_layout(pad=0.6)
    _save(fig, save_path)
    plt.show()


def plot_pal(study: dict, save_path="images/prompting/pal.png"):
    """Grouped bars: accuracy reasoning in prose vs writing and running a program.

    ``study`` carries labels and two accuracy lists (fractions in [0, 1]). The
    program-aided bar climbs where the problem is easy to translate to code but
    the arithmetic is gnarly enough that prose reasoning slips (the wins), ties at
    the ceiling on problems the model can already grind out in its head, and stays
    stuck where the model mis-models the problem so the program runs to a confident
    wrong number (the honest limit).
    """
    labels = study["labels"]
    cot = [100 * a for a in study["cot"]]
    pal = [100 * a for a in study["pal"]]
    x = np.arange(len(labels)); w = 0.36

    fig, ax = plt.subplots(figsize=(8, 4.2))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    _grid_ax(ax)
    bars = [ax.bar(x - w/2, cot, w, color=BLUE,   zorder=3, label="in prose"),
            ax.bar(x + w/2, pal, w, color=ORANGE, zorder=3, label="as a program")]
    for group in bars:
        for bar in group:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f"{bar.get_height():.0f}", ha="center", va="bottom",
                    fontsize=9.5, fontweight="600", color=DGRAY)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10, color=DGRAY)
    ax.set_ylabel(f"accuracy over {study['samples']} tries (%)",
                  fontsize=11.5, color=MGRAY)
    ax.set_ylim(0, 118); ax.set_yticks([0, 25, 50, 75, 100])
    ax.legend(fontsize=10.5, framealpha=0, loc="upper center", ncol=2)
    ax.set_title("Program-Aided LMs: Let the Interpreter Do the Arithmetic",
                 fontsize=12.5, fontweight="600", color=DGRAY, pad=8)
    plt.tight_layout(pad=0.6)
    _save(fig, save_path)
    plt.show()


def plot_cove(study: dict, save_path="images/augmentation/cove.png"):
    """Two panels: claim precision and recall, before (draft) and after (CoVe).

    ``study`` carries labels and four lists (fractions in [0, 1]): draft/cove
    precision and draft/cove recall. The two panels have to be read together. CoVe
    raises precision where the draft over-reaches and the factored check is sound
    (Portugal, Mexico) while leaving recall alone; where the model is confidently
    wrong on the check itself (Brazil) precision cannot climb (the draft was already
    precise) and recall instead collapses -- the boundary where retrieval, not more
    self-questioning, is the real fix.
    """
    labels = study["labels"]
    x = np.arange(len(labels)); w = 0.36
    panels = [("Precision: are the listed items correct?",
               study["draft_precision"], study["cove_precision"]),
              ("Recall: did we keep the correct ones?",
               study["draft_recall"], study["cove_recall"])]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    fig.patch.set_facecolor("white")
    for ax, (title, before, after) in zip(axes, panels):
        _grid_ax(ax)
        b = [100 * a for a in before]; a2 = [100 * a for a in after]
        bars = [ax.bar(x - w/2, b,  w, color=BLUE,   zorder=3, label="draft"),
                ax.bar(x + w/2, a2, w, color=ORANGE, zorder=3, label="after CoVe")]
        for group in bars:
            for bar in group:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                        f"{bar.get_height():.0f}", ha="center", va="bottom",
                        fontsize=9, fontweight="600", color=DGRAY)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10, color=DGRAY)
        ax.set_ylim(0, 118); ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_title(title, fontsize=11.5, fontweight="600", color=DGRAY, pad=6)
    axes[0].set_ylabel(f"percent (over {study['samples']} tries)",
                       fontsize=11, color=MGRAY)
    handles, leg_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, leg_labels, fontsize=10, framealpha=0, ncol=2,
               loc="lower center", bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Chain-of-Verification: Buying Precision, Sometimes With Recall",
                 fontsize=13.5, fontweight="600", color=DGRAY, y=1.02)
    plt.tight_layout(pad=1.0, rect=(0, 0.06, 1, 1))
    _save(fig, save_path)
    plt.show()


def plot_crag(study: dict, save_path="images/augmentation/crag.png"):
    """Grouped bars: answer accuracy with plain RAG vs with corrective RAG, per query.

    ``study`` carries labels and two accuracy lists (fractions in [0, 1]). On the
    queries the local corpus can answer, the retrieval evaluator grades CORRECT and
    CRAG ties the baseline, taking no web detour and doing no harm. On the trap queries
    the corpus cannot answer, plain RAG either begs off or makes something up while CRAG
    grades the retrieval INCORRECT, falls back to a web search, and recovers the fact.
    The whole gap is the traps, which is the point: correction buys robustness to
    retrieval failure, not omniscience.
    """
    labels = study["labels"]
    base = [100 * a for a in study["standard"]]
    crag = [100 * a for a in study["crag"]]
    x = np.arange(len(labels)); w = 0.36

    fig, ax = plt.subplots(figsize=(8, 4.2))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    _grid_ax(ax)
    bars = [ax.bar(x - w/2, base, w, color=BLUE,   zorder=3, label="standard RAG"),
            ax.bar(x + w/2, crag, w, color=ORANGE, zorder=3, label="corrective RAG")]
    for group in bars:
        for bar in group:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f"{bar.get_height():.0f}", ha="center", va="bottom",
                    fontsize=9.5, fontweight="600", color=DGRAY)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10, color=DGRAY)
    ax.set_ylabel(f"answers with the correct fact (% of {study['samples']})",
                  fontsize=11.5, color=MGRAY)
    ax.set_ylim(0, 118); ax.set_yticks([0, 25, 50, 75, 100])
    ax.legend(fontsize=10.5, framealpha=0, loc="upper center", ncol=2)
    ax.set_title("Corrective RAG: Hold Steady When Retrieval Works, Recover When It Fails",
                 fontsize=11.5, fontweight="600", color=DGRAY, pad=8)
    plt.tight_layout(pad=0.6)
    _save(fig, save_path)
    plt.show()


def plot_casting_summary(card: dict, save_path="images/agentic/casting_summary.png"):
    """The whole season on one card: models as rows, auditions as columns. Each
    column is shaded against its own field (calibration inverts, since claimed
    confidence on an impossible question should be low) and the best score in the
    column is starred. A dash means the model never read for that part. Mixed
    yardsticks on purpose: every column keeps the metric its own section judged
    by, so the card summarizes the chapter rather than inventing a new benchmark."""
    from matplotlib.colors import LinearSegmentedColormap
    models, cols = card["models"], card["columns"]
    n_r, n_c = len(models), len(cols)
    shade = LinearSegmentedColormap.from_list("card", ["#F7F9FE", BLUE])

    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.set_xlim(-1.75, n_c + 0.05); ax.set_ylim(n_r + 0.9, -1.75); ax.axis("off")

    for j, col in enumerate(cols):
        vals = col["values"]
        present = [vals[m] for m in models if m in vals]
        lo, hi = min(present), max(present)
        best = lo if col["lower_better"] else hi
        for i, m in enumerate(models):
            if m not in vals:                       # never auditioned for this part
                ax.add_patch(mp.Rectangle((j, i), 1, 1, facecolor=LGRAY,
                                          edgecolor="white", lw=1.5))
                ax.text(j + 0.5, i + 0.5, "—", ha="center", va="center",
                        fontsize=10, color=MGRAY)
                continue
            v = vals[m]
            t = 0.5 if hi == lo else (v - lo) / (hi - lo)
            t = 1 - t if col["lower_better"] else t
            ax.add_patch(mp.Rectangle((j, i), 1, 1, facecolor=shade(t),
                                      edgecolor="white", lw=1.5))
            star = v == best
            ax.text(j + 0.5, i + 0.5, ("★" if star else "") + col["fmt"].format(v),
                    ha="center", va="center", fontsize=9.5,
                    fontweight="700" if star else "500",
                    color="white" if t > 0.55 else DGRAY)
        ax.text(j + 0.45, -0.18, col["label"], ha="left", va="bottom",
                fontsize=9.5, fontweight="600", color=DGRAY, rotation=24)
    for i, m in enumerate(models):
        ax.text(-0.12, i + 0.5, short_model(m), ha="right", va="center",
                fontsize=10.5, fontweight="700", color=model_color(m))
    has_dash = any(m not in col["values"] for col in cols for m in models)
    legend = "★ best score in the column"
    if has_dash:                                   # only explain the dash if one shows
        legend += "   ·   — did not audition"
    legend += ("   ·   calibration = confidence claimed on impossible questions "
               "(1-5, lower is better)")
    ax.text(0, n_r + 0.42, legend,
            ha="left", va="top", fontsize=8.5, color=MGRAY)
    ax.set_title("Who Won What? The Season on One Card",
                 fontsize=13, fontweight="600", color=DGRAY, pad=8)
    plt.tight_layout(pad=0.6)
    _save(fig, save_path)
    plt.show()


def plot_spelling_survey(study, save_path="images/tokens/spelling_survey.png"):
    """A letter-counting survey as a grid: tricky words down the rows, models
    across the columns, each cell the model's single greedy answer. Green where it
    matches the true count in the left column, soft red where it misses, with each
    model's tally along the bottom. Most of the grid is red, because the repeated
    letters stay sealed inside chunks no model can see into, so more capability
    buys confident wrong answers rather than correct ones."""
    rows, models = study["rows"], study["models"]
    n_r, n_c = len(rows), len(models)
    miss_fill = "#FBE3E4"

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.set_xlim(-2.7, n_c + 1.05); ax.set_ylim(n_r + 0.85, -1.0); ax.axis("off")

    for i, row in enumerate(rows):
        true = row["true"]
        ax.text(-0.16, i + 0.5, f"{row['letter']} in {row['word']}", ha="right",
                va="center", fontsize=10.5, fontweight="600", color=DGRAY)
        ax.add_patch(mp.Rectangle((0, i), 1, 1, facecolor=LGRAY,
                                  edgecolor="white", lw=2.5))
        ax.text(0.5, i + 0.5, str(true), ha="center", va="center",
                fontsize=11.5, fontweight="700", color=DGRAY)
        for j, ans in enumerate(row["answers"]):
            ok = ans == true
            ax.add_patch(mp.Rectangle((j + 1, i), 1, 1,
                                      facecolor=GREEN if ok else miss_fill,
                                      edgecolor="white", lw=2.5))
            ax.text(j + 1.5, i + 0.5, "?" if ans is None else str(ans),
                    ha="center", va="center", fontsize=11.5,
                    fontweight="700" if ok else "500",
                    color="white" if ok else RED)

    ax.text(0.5, -0.16, "true", ha="center", va="bottom", fontsize=10,
            fontweight="700", color=MGRAY)
    for j, m in enumerate(models):
        ax.text(j + 1.5, -0.16, short_model(m), ha="center", va="bottom",
                fontsize=10, fontweight="700", color=model_color(m))
        hits = sum(r["answers"][j] == r["true"] for r in rows)
        ax.text(j + 1.5, n_r + 0.12, f"{hits}/{n_r}", ha="center", va="top",
                fontsize=9.5, fontweight="600", color=MGRAY)
    ax.text(-0.16, n_r + 0.12, "correct", ha="right", va="top",
            fontsize=9.5, fontstyle="italic", color=MGRAY)
    ax.set_title("Counting Letters the Tokenizer Hides", fontsize=12.5,
                 fontweight="600", color=DGRAY, pad=12)
    plt.tight_layout(pad=0.5)
    _save(fig, save_path)
    plt.show()


def plot_plan(study: dict, save_path="images/agentic/plan.png"):
    """One bar per strategy: the share of plans that come out fully valid under
    every dependency after a new constraint arrives.

    Editing the draft in place satisfies the new rule but regresses on an old one
    about half the time; re-deriving the plan from the full constraint set is far
    better but still fumbles a fresh sort one time in five; only re-checking the
    whole plan and bouncing each broken edge back reaches every-time validity. The
    bars climb left to right from patch to rebuild to checked.
    """
    labels = study["labels"]
    vals = [100 * v for v in study["valid"]]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(7, 4.2))
    fig.patch.set_facecolor("white")
    _grid_ax(ax)
    bars = ax.bar(x, vals, 0.55, color=[ORANGE, BLUE, GREEN], zorder=3)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f"{bar.get_height():.0f}", ha="center", va="bottom",
                fontsize=10.5, fontweight="600", color=DGRAY)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11, color=DGRAY)
    ax.set_ylabel(f"plans fully valid over {study['samples']} runs (%)",
                  fontsize=11.5, color=MGRAY)
    ax.set_ylim(0, 112); ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_title("Keeping a Plan Valid: Edit, Re-derive, or Check",
                 fontsize=12.5, fontweight="600", color=DGRAY, pad=8)
    plt.tight_layout(pad=0.6)
    _save(fig, save_path)
    plt.show()


# ── Dispatch contest: over- and under-routing against the golden answer ───────
def _router_label(m: str) -> str:
    """Keep the size tag on the gemma4 router so it reads apart from the e2b/26b
    resolvers it shares a family with; shorten everyone else."""
    return m if m.startswith("gemma4") else short_model(m)


def plot_dispatch(picks, tasks, save_path="images/agentic/dispatch_contest.png"):
    """Each router's twelve picks scored against the gold tier, one stacked bar per
    model: under-routed (sent too weak, the task fails -- a quality cost, red, on
    the left), correct (right size, green), over-routed (sent too big, solved but
    wasteful -- a time cost, amber, on the right). Sorted by correct with the best
    dispatcher on top, so the ideal router is almost all green, the lazy all-HARD
    router all amber, the weak ones all red. ``picks`` is ``{model: {task_id:
    tier}}``; ``tasks`` carry the gold ``tier``."""
    rank = {"easy": 0, "medium": 1, "hard": 2}
    gold = {t["id"]: rank[t["tier"]] for t in tasks}
    n = len(tasks)

    def split(p):
        u = c = o = 0
        for tid, tier in p.items():
            d = rank[tier] - gold[tid]
            u, c, o = (u + 1, c, o) if d < 0 else (u, c + 1, o) if d == 0 else (u, c, o + 1)
        return u, c, o

    rows = {m: split(p) for m, p in picks.items()}
    models = sorted(rows, key=lambda m: (rows[m][1], -rows[m][0]))   # best ends on top
    y = np.arange(len(models))

    fig, ax = plt.subplots(figsize=(8, 0.5 * len(models) + 1.5))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")

    left = np.zeros(len(models))
    for k, col in [(0, RED), (1, GREEN), (2, AMBER)]:
        vals = np.array([rows[m][k] for m in models])
        ax.barh(y, vals, left=left, height=0.62, color=col, zorder=3,
                edgecolor="white", linewidth=1.4)
        for yi, (v, l) in enumerate(zip(vals, left)):
            if v:
                ax.text(l + v / 2, yi, str(int(v)), ha="center", va="center",
                        fontsize=10, fontweight="700", color="white")
        left += vals

    ax.set_yticks(y); ax.set_yticklabels([_router_label(m) for m in models],
                                         fontsize=10.5, color=DGRAY)
    ax.set_xlim(0, n); ax.set_xticks(range(0, n + 1, 2))
    ax.tick_params(colors=MGRAY, labelsize=10)
    ax.set_xlabel("tasks routed (of 12)", fontsize=11, color=MGRAY)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(LGRAY)

    handles = [mp.Patch(color=c, label=l) for l, c in
               (("under-routed (fails)", RED), ("correct", GREEN),
                ("over-routed (wasteful)", AMBER))]
    ax.legend(handles=handles, ncol=3, fontsize=9.5, frameon=False,
              loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax.set_title("The dispatcher contest, scored against the golden routing",
                 fontsize=13, fontweight="600", color=DGRAY, pad=30)

    plt.tight_layout(pad=0.8)
    _save(fig, save_path)
    plt.show()
