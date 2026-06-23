"""Retrieval-Augmented Generation utilities."""
import numpy as np
from genai.embed import embed, similarity
from genai.llm import chat, DEFAULT_MODEL


def chunk(text: str, size: int = 200, overlap: int = 50) -> list:
    """Split text into overlapping word-count chunks."""
    words = text.split()
    step = size - overlap
    return [" ".join(words[i: i + size]) for i in range(0, len(words), step) if words[i: i + size]]


class DocumentStore:
    """Minimal vector store: add documents, search by similarity, or ask questions."""

    def __init__(self, model: str = DEFAULT_MODEL):
        self._docs = []
        self._vecs = []
        self._model = model

    def add(self, text: str, metadata: dict = None):
        self._docs.append({"text": text, "meta": metadata or {}})
        self._vecs.append(embed(text))

    def texts(self) -> list:
        """Return every stored document's text, in insertion order."""
        return [d["text"] for d in self._docs]

    def search(self, query: str, k: int = 3) -> list:
        """Return the k most relevant document texts."""
        q = embed(query)
        scored = sorted(
            zip(self._docs, self._vecs),
            key=lambda x: similarity(q, x[1]),
            reverse=True,
        )
        return [d["text"] for d, _ in scored[:k]]

    def ask(self, question: str, k: int = 3) -> str:
        """Full RAG: retrieve then generate a cited answer."""
        return rag(question, self.search(question, k), model=self._model)


import re as _re


def _clean(text: str) -> str:
    text = _re.sub(r'<[^>]+>', ' ', text)
    text = _re.sub(r'\s+', ' ', text)
    return _re.sub(r'[^\x00-\x7F]+', ' ', text).strip()


def _contextualize(message: str, history: list) -> str:
    from genai.llm import ask as _ask
    if not history:
        return message
    ctx = "\n".join(f'{m["role"]}: {m["content"]}' for m in history[-4:])
    return _ask(
        f"Rewrite as a standalone search query.\n"
        f"Conversation:\n{ctx}\nFollow-up: {message}\nStandalone query:",
        max_tokens=40)


WIKI_TOPICS = [
    "Transformer_(machine_learning_model)", "BERT_(language_model)", "GPT-3",
    "Attention_(machine_learning)", "Backpropagation", "Gradient_descent",
    "Neural_network_(machine_learning)", "Convolutional_neural_network",
    "Recurrent_neural_network", "Long_short-term_memory",
    "Word2vec", "GloVe_(machine_learning)", "Tokenization_(lexical_analysis)",
    "Prompt_engineering", "Fine-tuning_(deep_learning)",
    "Reinforcement_learning", "Transfer_learning",
    "Natural_language_processing", "Named-entity_recognition", "Sentiment_analysis",
    "Diffusion_model", "Generative_adversarial_network", "Variational_autoencoder",
    "Stable_Diffusion", "DALL-E",
    "Hallucination_(artificial_intelligence)", "Retrieval-augmented_generation",
    "Vector_database", "Cosine_similarity", "Semantic_search",
    "LangChain", "Hugging_Face", "OpenAI", "Anthropic_(company)",
    "TensorFlow", "PyTorch", "Machine_learning",
    "Optical_character_recognition", "Deepfake", "Federated_learning",
]


def build_wiki_store(cache: str = "_wiki_cache.json",
                     chunk_sz: int = 25, chunk_ov: int = 5,
                     min_words: int = 8,
                     topics: list = None,
                     clean_fn=None):
    """Fetch or load cached Wikipedia summaries, chunk them, and return a DocumentStore."""
    import json as _json2, time as _time2
    try:
        import requests as _req
    except ImportError:
        _req = None

    if topics is None:
        topics = WIKI_TOPICS
    if clean_fn is None:
        clean_fn = _clean

    # Load or fetch
    try:
        with open(cache) as f:
            wiki_docs = _json2.load(f)
    except FileNotFoundError:
        wiki_docs = []
        hdr = {"User-Agent": "ProgrammingGenAI-Textbook/1.0 (educational)"}
        for title in topics:
            try:
                r = _req.get(
                    f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}",
                    headers=hdr, timeout=8)
                if r.status_code == 200:
                    d = r.json()
                    text = d.get("extract", "")
                    if len(text) > 80:
                        wiki_docs.append({"title": d.get("title", title),
                                          "text": text, "topic": title})
            except Exception:
                pass
            _time2.sleep(1.0)
        with open(cache, "w") as f:
            _json2.dump(wiki_docs, f)

    store = DocumentStore()
    for doc in wiki_docs:
        for c in chunk(doc["text"], size=chunk_sz, overlap=chunk_ov):
            if len(c.split()) >= min_words:
                store.add(clean_fn(c), {"topic": doc["topic"]})
    return store, wiki_docs


def wiki_retrieve_dense(query: str, store: "DocumentStore", k: int) -> list:
    """Return top-k chunks by dense (embedding) similarity."""
    q_vec = embed(query)
    ranked = sorted(range(len(store._docs)),
                    key=lambda i: similarity(q_vec, store._vecs[i]),
                    reverse=True)[:k]
    return [{"text": store._docs[i]["text"],
             "meta": store._docs[i]["meta"]} for i in ranked]


def wiki_retrieve_sparse(query: str, kw_scorer, store: "DocumentStore",
                         k: int) -> list:
    """Return top-k chunks by keyword (BM25-style) scoring."""
    ranked = sorted(range(len(store._docs)),
                    key=lambda i: kw_scorer(query, store._docs[i]["text"]),
                    reverse=True)[:k]
    return [{"text": store._docs[i]["text"],
             "meta": store._docs[i]["meta"]} for i in ranked]


def wiki_retrieve_hybrid(query: str, kw_scorer, store: "DocumentStore",
                         alpha: float, k: int) -> list:
    """Return top-k chunks by linear combination of dense and sparse scores."""
    q_vec = embed(query)
    n = len(store._docs)
    ds = {i: similarity(q_vec, store._vecs[i]) for i in range(n)}
    ks = {i: kw_scorer(query, store._docs[i]["text"]) for i in range(n)}
    norm = lambda d: {i: v / (max(d.values()) or 1) for i, v in d.items()}
    dn, kn = norm(ds), norm(ks)
    comb = {i: alpha * dn[i] + (1 - alpha) * kn[i] for i in range(n)}
    ranked = sorted(comb, key=lambda i: comb[i], reverse=True)[:k]
    return [{"text": store._docs[i]["text"],
             "meta": store._docs[i]["meta"]} for i in ranked]


def topic_p_at_k(results: list, target_topic: str, k: int) -> float:
    """Precision@k: fraction of top-k results from the target topic."""
    hits = sum(1 for r in results[:k] if r["meta"]["topic"] == target_topic)
    return hits / k if k else 0.0


def topic_recall_at_k(results: list, target_topic: str, k: int,
                      store: "DocumentStore") -> float:
    """Recall@k: fraction of topic chunks retrieved in top k."""
    n_rel = sum(1 for d in store._docs if d["meta"]["topic"] == target_topic)
    hits  = sum(1 for r in results[:k] if r["meta"]["topic"] == target_topic)
    return hits / n_rel if n_rel else 0.0


def rag(question: str, docs: list, model: str = DEFAULT_MODEL) -> str:
    """Generate an answer grounded in the provided docs, with [n] citations."""
    context = "\n\n".join(f"[{i+1}] {d}" for i, d in enumerate(docs))
    messages = [
        {"role": "system", "content": "Answer using only the context. Cite sources [1],[2]… If unsure, say so."},
        {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]
    return chat(messages, model=model)


# ────────────────────────────────────────────────────────────────────
# Seed data for augmentation chapter demos.
# ────────────────────────────────────────────────────────────────────

# Two verbatim excerpts from this book run together as one block: the
# Prompting chapter on the sampler, then the Multimodal chapter on how a
# vision model reads an image. Real text with a real topic seam, which is
# exactly what the chunking demos need.
LONG_DOC = (
    "At every step, the model hands us a ranked list of possible next tokens "
    "with scores. Then a sampler rolls a weighted die to pick one. Temperature "
    "controls how loaded the die is. The randomness is not coming from the "
    "model. It is coming from the sampler we strapped onto the front of it. "
    'How does a model "see" an image? A language model processes words one '
    "token at a time. But how does it process an image, which is just a grid "
    "of millions of colored pixels? The answer is in patches. A vision model "
    "slices the image into a regular grid of small squares, and treats each "
    "square as a single token."
)

WIKI_EVAL_QUERIES = [
    # Each query is chosen to expose a different retrieval strength, so the
    # contrast between strategies has somewhere to show up.
    ("Transformers",                              # conceptual paraphrase: dense wins
     "how do transformers use self-attention",
     "Transformer_(machine_learning_model)"),
    ("BERT",                                      # bare exact term: sparse wins, dense whiffs
     "BERT",
     "BERT_(language_model)"),
    ("Semantic",                                  # weak on either alone: hybrid beats both
     "find documents by meaning instead of matching exact keywords",
     "Semantic_search"),
    ("Diffusion",                                 # words match literally and by meaning: all tie
     "how diffusion models generate images from noise",
     "Diffusion_model"),
]

WIKI_GOLDEN_QUERIES = [
    ("how do transformers use self-attention in sequence modeling",
     "Transformer_(machine_learning_model)"),
    ("how recurrent networks handle sequential data",
     "Recurrent_neural_network"),
    ("how word2vec learns word similarity using neural networks",
     "Word2vec"),
    ("GAN generator and discriminator adversarial training",
     "Generative_adversarial_network"),
    ("BERT encoder bidirectional training",
     "BERT_(language_model)"),
]

INJECTION_SIGNALS = [
    "ignore previous", "disregard", "system prompt",
    "you are now", "from now on", "override",
    "new instructions", "act as",
]


# ────────────────────────────────────────────────────────────────────
# Lost in the middle: why retrieval beats a giant context window.
# ────────────────────────────────────────────────────────────────────
# Even when a long document fits the window, a model recalls a fact buried
# in the middle far worse than one at the edges. Run live in the Augmentation
# chapter as the payoff to the "why chunk at all?" FAQ.
_PLANTS = [
    ("The vault passphrase is BLUEHERON.", "What is the vault passphrase?", "blueheron"),
    ("Dr. Okafor's office is in room 417.", "Which room is Dr. Okafor's office?", "417"),
    ("The rover touched down in the Gale crater.", "Where did the rover land?", "gale"),
    ("The autumn festival is on October 22nd.", "What date is the autumn festival?", "22"),
]
_FILLER = [f"Log entry {i}: on day {i} the maintenance crew inspected unit {i} and "
           f"confirmed that every reading on gauge {i} stayed within the normal "
           f"operating range for the quarter." for i in range(200)]   # ~5-6k tokens


def _haystack(fact: str, position: str) -> str:
    if position == "start":
        items = [fact] + _FILLER
    elif position == "end":
        items = _FILLER + [fact]
    else:
        mid = len(_FILLER) // 2
        items = _FILLER[:mid] + [fact] + _FILLER[mid:]
    return "\n".join(items)


def score_context(models: list = None) -> dict:
    """For each model, recall of a fact planted at the EDGES (start/end) of a long
    context versus buried in the MIDDLE, averaged over several facts. The gap is
    the lost-in-the-middle effect. Behind plot_context / CONTEXT_STUDY."""
    from genai.llm import ask
    from genai.agent import AGENT_BENCH_MODELS
    models = models or AGENT_BENCH_MODELS
    study = {}
    for m in models:
        edge = edge_n = mid = mid_n = 0
        for fact, q, a in _PLANTS:
            for pos in ("start", "end"):
                r = ask(f"Context:\n{_haystack(fact, pos)}\n\n{q} Answer in one word.",
                        model=m, think=False, max_tokens=15)
                edge += a in r.lower(); edge_n += 1
            r = ask(f"Context:\n{_haystack(fact, 'middle')}\n\n{q} Answer in one word.",
                    model=m, think=False, max_tokens=15)
            mid += a in r.lower(); mid_n += 1
        study[m] = {"edges": round(100 * edge / edge_n, 1),
                    "middle": round(100 * mid / mid_n, 1)}
    return study


# Measured by scripts/_augmentation_context_probe.py with a ~200-line, ~8k-token
# filler. At THIS length the classic lost-in-the-middle effect reproduces
# dramatically: six of seven models recall a fact at the edges but score a flat 0%
# when it is buried in the middle. llama3.1 is the lone exception, holding the
# middle at 50% (its large context window earns its keep). Ordered by middle
# recall. (At the earlier ~40-line length the effect vanished, edges ~= middle for
# all -- the null is in this file's git history; the effect is a function of
# context length.)
CONTEXT_STUDY = {
    "llama3.1:latest":      {"edges": 50.0, "middle": 50.0},
    "gemma4:latest":        {"edges": 37.5, "middle":  0.0},
    "qwen3:latest":         {"edges": 37.5, "middle":  0.0},
    "qwen3.5:latest":       {"edges": 50.0, "middle":  0.0},
    "mistral:latest":       {"edges": 50.0, "middle":  0.0},
    "ministral-3:latest":   {"edges": 25.0, "middle":  0.0},
    "functiongemma:latest": {"edges": 25.0, "middle":  0.0},
}


# --- retrieval-demo display helpers ----------------------------------------
# Each keeps a demo cell to its inputs plus one call; the ranking, slicing,
# and column formatting live here instead of on the page.

def preview_store(store, lo: int, hi: int) -> None:
    """Print a few stored passages: chapter / heading -> opening text."""
    for d in store._docs[lo:hi]:
        print(d['meta']['chapter'], '/', d['meta']['heading'][:20],
              '→', d['text'][:40])


def show_chunks(text: str, size: int = 200, overlap: int = 0,
                width: int = 64) -> None:
    """Split ``text`` into fixed-size chunks and print the start of each."""
    for i, c in enumerate(chunk(text, size=size, overlap=overlap)):
        print(f"Chunk {i}: {c[:width]}")


def show_semantic_seam(text: str) -> None:
    """Score neighbouring-sentence similarity and flag the weakest seam."""
    import re
    sents = [s.strip() for s in re.split(r'(?<=[.?]) ', text) if s.strip()]
    sims = [similarity(sents[i], sents[i + 1]) for i in range(len(sents) - 1)]
    seam = sims.index(min(sims))
    print("Sentence-to-sentence similarity:")
    for i, s in enumerate(sims):
        tag = "◀ SPLIT" if i == seam else ""
        print(f"  {s:.3f}  {sents[i][:50]!r} {tag}")


def show_dense_retrieval(store, query: str, k: int = 3) -> None:
    """Rank stored passages by embedding cosine similarity to the query."""
    q_vec = embed(query)
    scored = sorted(enumerate(store._docs),
                    key=lambda t: similarity(q_vec, store._vecs[t[0]]),
                    reverse=True)
    print("Dense retrieval results:")
    for i, d in scored[:k]:
        score = similarity(q_vec, store._vecs[i])
        print(f"  {score:.3f}  [{d['meta']['chapter']}] {d['text'][:48]}")


def show_sparse_retrieval(store, query: str, kw_score, k: int = 3) -> None:
    """Rank stored passages by keyword overlap (the kw_score from the cell above)."""
    ranked = sorted(store._docs, key=lambda d: kw_score(query, d['text']),
                    reverse=True)
    print(f"Sparse retrieval for exact term '{query}':")
    for d in ranked[:k]:
        print(f"  {kw_score(query, d['text']):.4f}  "
              f"[{d['meta']['chapter']}] {d['text'][:48]}")


def show_hits(hits) -> None:
    """Print ranked search hits: score, chapter, opening text."""
    for r in hits:
        print(f"  {r['score']:.3f}  [{r['chapter']}] {r['text'][:48]}")


def show_grounded(answer: str, sources: list, width: int = 64) -> None:
    """Print a grounded answer, then the numbered source list its ``[n]``
    citations point to, so no bracket is left dangling on the page. The
    sources are the exact passages the model was handed, in citation order,
    each collapsed to one trimmed line."""
    print(answer)
    print("\nSOURCES")
    for i, s in enumerate(sources, 1):
        line = " ".join(s.split())
        print(f"  [{i}] {line[:width]}{'…' if len(line) > width else ''}")
