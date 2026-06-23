# Programming Generative AI — companion code

This is the companion code repository for the book *Programming Generative AI*. It holds the `genai` helper library that every chapter imports, and the `scripts/` that reproduce the book's figures and transcripts. The book's prose and notebooks live separately; this repo is just the code.

All of the examples run on small open models through [Ollama](https://ollama.com), so you can reproduce everything on your own machine, with no API keys and no cloud bill.

## Setup

Install [Ollama](https://ollama.com) and [uv](https://docs.astral.sh/uv/), then:

```bash
uv sync
ollama pull gemma4
ollama pull qwen2.5-coder
ollama pull nomic-embed-text
```

The default chat model is `gemma4:latest`, code tasks use `qwen2.5-coder:latest`, and embeddings come from `nomic-embed-text`.

## The `genai` library

The code in the book is deliberately short: no example runs past about fifteen lines, and the machinery that makes that possible lives in [`genai/`](genai/). When a chapter calls `ask(...)`, `rag(...)`, or `show_judge(...)`, the helper is defined here. Everything imports from one place:

```python
from genai import ask, embed, similarity, DocumentStore, agent, think
```

Three modules carry the whole book and are worth knowing from the start:

- [`genai/llm.py`](genai/llm.py): `ask` and `chat`, the two calls everything else builds on. Every LLM call enforces concise output by default (a brevity system prompt plus `max_tokens=200`) so outputs stay readable; pass `system=None` and a larger `max_tokens` when you want room.
- [`genai/embed.py`](genai/embed.py): `embed`, `similarity`, `semantic_search`, and `word_analogy`, the embedding toolkit reused by retrieval, memory, and routing demos across the book.
- [`genai/viz.py`](genai/viz.py): every `plot_*` figure in the book, one helper per chart, all drawn from measured data.

The remaining modules belong to specific chapters (`rag.py`, `crag.py`, `cove.py`, `agent.py`, `thinking.py`, `prompting.py`, `pal.py`, `security.py`, `privacy.py`, and others).

## Reproducers

Every measured demo in the book keeps a committed reproducer in [`scripts/`](scripts/) — the `_*_probe.py` and `_*_capture.py` files regenerate the numbers and transcripts a figure is built from. The remaining scripts are book-production tooling (notebook execution, output trimming, PDF build) and reference the book's notebooks directly.

## License

MIT — see [LICENSE](LICENSE).
