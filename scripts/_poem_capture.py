#!/usr/bin/env python3
"""Reproducer for the preface poem (front.ipynb, cell front-sophia-poem).

gemma4 (the model that plays Sophia) was handed the book's table of contents
with one-phrase gists and asked for a free-verse poem, one line per chapter.
A fresh batch was rolled on 2026-06-18 against the current 12-chapter spine
(Foundations folded into Intro, Design dropped, Autonomy added); roll 3 (baked
below) was chosen and placed in the preface verbatim. Re-running prints a fresh
roll for comparison; the poem in the book stays the chosen roll.

Run: .venv/bin/python scripts/_poem_capture.py
"""
from genai.llm import ask

GIST = """The book: "Programming Generative AI" by Chike Abuah. Chapters in order:
1. Introduction - the bet: learn generative AI by building everything yourself on local models; how language models work, probability and predicting the next token
2. Prompting - the craft of asking well; temperature, top-k, the control panel
3. Tokens - the units a model actually reads; byte pair encoding
4. Semantics - embeddings; meaning becomes geometry, similar things land near each other
5. Metacoding - models that write and read code
6. Augmentation - retrieval (RAG); teaching the model to look things up before answering
7. Agentic - giving the model hands: tools, memory, actions; Sophia the book's assistant
8. Autonomy - deploying agents safely; observability, computer use, a library of skills
9. Thinking - reasoning models that think before they answer
10. Multimodal - images and audio; models that see and hear
11. Responsible - security, privacy, what goes wrong and how to defend
12. Efficiency - quantization and speed; doing more with less"""

PROMPT = (GIST + "\n\nWrite a short free-verse poem about this book's "
          "journey: one line per chapter, in chapter order, 12 lines. No "
          "rhyme needed; aim for vivid, concrete images. Each line under "
          "12 words. No titles or numbering, just the 12 lines.")

CHOSEN = """Building brains, token by next guess.
Crafting whispers with careful dials.
Invisible slices, the model's breath.
Meaning mapped onto colored space.
Code appearing from pure language.
Tethering knowledge to retrieved facts.
Hands extending: tools and recall.
Watching the agent walk through the world.
Thought unfolding, layer by layer thought.
Pixels bloom, then sound echoes forth.
Guardrails built against the unseen fray.
Shrinking giants to fit the small core."""

if __name__ == "__main__":
    print("=== the chosen roll (in the preface) ===")
    print(CHOSEN)
    print("\n=== a fresh roll for comparison ===")
    print(ask(PROMPT, model="gemma4:latest", system=None, max_tokens=400,
              think=False).strip())
