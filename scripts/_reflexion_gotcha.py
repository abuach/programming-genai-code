"""THROWAWAY prototype v3: 'gotcha' coding tasks for the Reflexion demo.

Each task has an EASY core the model writes correctly every time, plus ONE
overlooked requirement (empty input, order preservation, dedup, whitespace) that
the model systematically omits on the first try and on resamples (a stable wrong
prior, so blind retry stays stuck), but can add in one line once a failing test
names it (so reflection's lesson unlocks it). This is the band where Reflexion
should beat blind retry.

Reuses the actor/evaluator/reflect/run_condition harness from _reflexion_proto.
Run: .venv/bin/python scripts/_reflexion_gotcha.py [model] [repeats]
"""
import sys
sys.path.insert(0, "scripts")
sys.path.insert(0, "chapters")
import _reflexion_proto as P

# Specs are deliberately terse, like a real informal request. The TEST CASES
# encode the full contract (the edge cases a first attempt rarely anticipates).
# The agent discovers the missing requirement from the failing test and reflects
# on it -- which is exactly Reflexion's premise. The comment after each task names
# the hidden gotcha and the stable wrong prior it provokes.
# Re-themed as a book-assistant's toolkit (the work Sophia would actually do for a
# reader) without changing any hidden contract: each spec hides the SAME edge case
# its generic ancestor did, so the demo still measures reflection, not vocabulary.
GOTCHA_SUITE = [
    ("nightly_pace",                                     # empty list -> ZeroDivision
     "Write nightly_pace(counts): return the mean of the page counts in the list.",
     [(([1, 2, 3],), 2.0), (([10],), 10.0), (([],), 0.0), (([2, 4],), 3.0)]),
    ("bookmark_index",                                   # not found -> None/raise
     "Write bookmark_index(bookmarks, page): return the position of page in the "
     "bookmarks list.",
     [(([1, 2, 3], 2), 1), (([1, 2, 3], 9), -1), (([], 5), -1), (([7], 7), 0)]),
    ("unique_books",                                     # set() loses order
     "Write unique_books(reading_list): remove the duplicate books from reading_list.",
     [(([3, 1, 3, 2, 1],), [3, 1, 2]), (([1, 2, 3],), [1, 2, 3]),
      (([],), []), (([5, 5, 5],), [5])]),
    ("note_words",                                       # split(' ') on '' -> 1
     "Write note_words(note): count the number of words in the reader's note.",
     [(("hello world",), 2), (("",), 0), (("   ",), 0), (("  a  b c ",), 3)]),
    ("author_initials",                                  # split(' ') eats blanks
     "Write author_initials(name): return the uppercase initials of the words in "
     "the author's name.",
     [(("ada lovelace",), "AL"), (("  grace   hopper ",), "GH"), (("alan",), "A")]),
    ("runner_up",                                        # forgets DISTINCT
     "Write runner_up(scores): return the second highest quiz score in scores.",
     [(([5, 5, 3],), 3), (([1, 2, 3, 4],), 3), (([10, 8, 10, 8],), 8),
      (([7, 3],), 3)]),
    ("format_title",                                     # .title() keeps spaces
     "Write format_title(title): capitalize each word in the book title.",
     [(("hello   world",), "Hello World"), (("a b",), "A B"), (("",), "")]),
    ("rotate_picks",                                     # k > len, k=0 wrap
     "Write rotate_picks(books, k): rotate the daily book recommendations to the "
     "right by k places.",
     [(([1, 2, 3, 4], 1), [4, 1, 2, 3]), (([1, 2, 3], 0), [1, 2, 3]),
      (([1, 2, 3], 4), [3, 1, 2]), (([], 2), [])]),
    ("round_rating",                                     # round() is banker's: 2.5->2
     "Write round_rating(score): round a star score to the nearest whole star, "
     "rounding a half up.",
     [((2.5,), 3), ((3.5,), 4), ((0.5,), 1), ((4.5,), 5), ((2.4,), 2), ((2.6,), 3)]),
    ("rounded_rating",                                   # mean THEN half-up round
     "Write rounded_rating(scores): average the star scores and round the mean to "
     "the nearest whole star, rounding a half up.",
     [(([5, 4],), 5), (([3, 4, 4],), 4), (([2, 3],), 3), (([1, 1, 1],), 1),
      (([4, 5, 5, 5],), 5)]),
    ("shelve_order",                                     # composite sort key (ties)
     "Write shelve_order(books): given (title, rating) pairs, return the titles "
     "sorted by rating from high to low, breaking ties by title alphabetically.",
     [(([("b", 5), ("a", 5), ("c", 3)],), ["a", "b", "c"]),
      (([("x", 1), ("y", 2)],), ["y", "x"]), (([],), []),
      (([("c", 3), ("a", 3), ("b", 3)],), ["a", "b", "c"])]),
]

P.SUITE = GOTCHA_SUITE  # run_condition / attempt read P.SUITE

if __name__ == "__main__":
    P.temp = 0.7
    model   = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5-coder:latest"
    repeats = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    K = 4
    print(f"model={model}  K={K}  repeats={repeats}  tasks={len(GOTCHA_SUITE)}",
          flush=True)
    refl  = P.run_condition(True,  model, K, 0.7, repeats, log=True)
    print("reflexion   by trial:", [f"{x:.0%}" for x in refl], flush=True)
    blind = P.run_condition(False, model, K, 0.7, repeats, log=True)
    print("blind retry by trial:", [f"{x:.0%}" for x in blind], flush=True)
