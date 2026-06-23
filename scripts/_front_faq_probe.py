#!/usr/bin/env python3
"""Reproducer for the front matter's "Ask Sophia" FAQ.

Re-runs the finished book assistant (full RAG over the committed book
snapshot) on the reader questions printed in the preface, so each frozen
cell's transcript can be regenerated and sanity checked. The answers make
live gemma4 calls and will vary run to run; what should survive is the shape:
grounded, cited, and honest about what the book does not cover (the
train-from-scratch question).

Run from chapters/:  ../.venv/bin/python ../scripts/_front_faq_probe.py
"""
from genai.book import BookIndex, show_faq

BookIndex().build()   # ensure the on-disk index exists (no-op once built)

# Cell A: questions a reader has before deciding to read the book.
print("===== A: About reading the book =====")
show_faq([
    "Do I need a math or computer science background to read this book?",
    "Why does the book use local open-source models instead of ChatGPT or Claude?",
    "Why do I get a slightly different answer each time I run the same code?",
])

# Cell B: questions that should resolve to a specific (sub)section, plus one
# the book deliberately does not cover (the train-from-scratch question).
print("===== B: Finding your way around the book =====")
show_faq([
    "Which section explains breaking a long document into pieces for retrieval?",
    "Where does the book teach stepping back to find the general principle before solving a problem?",
    "Where do I learn about the context window and running out of token space?",
    "Does the book teach me how to train a large language model from scratch?",
])
