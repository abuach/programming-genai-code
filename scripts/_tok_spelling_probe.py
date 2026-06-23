"""Regenerate genai.tokens.SPELLING_STUDY.

Asks each model to count a repeated letter in each tricky word once, at
temperature 0, and prints the study dict to paste back into genai/tokens.py.
Run from the chapters/ directory so `genai` is importable.
"""
import pprint

from genai.tokens import score_spelling_survey

if __name__ == "__main__":
    pprint.pp(score_spelling_survey())
