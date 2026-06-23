"""Screening probe: hunt for gotcha tasks that SPREAD the reflexion bake-off.

The 8-task GOTCHA_SUITE clusters the field (five models ace it 100/100, only
three show any reflexion gap). To spread the board we need tasks with a
*broadly shared* stable wrong prior: a trap even strong models walk into on the
first try (so the ceiling drops) but can fix from a one-line lesson (so reflexion
pulls ahead of blind retry). This screens candidate tasks across a strong/mid/
weak model spread and reports, per task per model, the blind try-1 pass rate, the
final blind@K rate, and the final reflexion@K rate. Keepers: try-1 low across the
field AND reflexion@K > blind@K for several models.

Run: .venv/bin/python scripts/_reflexion_spread.py [repeats] [model ...]
"""
import os
import sys
sys.path.insert(0, "scripts")
sys.path.insert(0, "chapters")
import _reflexion_proto as P

# Candidate tasks aimed at SPREAD. The spec is terse; the cases hold the contract.
# The comment names the shared wrong prior each one provokes.
CANDIDATES = [
    ("round_rating",                         # round() is banker's: round(2.5)==2
     "Write round_rating(score): round a star score to the nearest whole star, "
     "rounding a half up.",
     [((2.5,), 3), ((3.5,), 4), ((0.5,), 1), ((4.5,), 5), ((2.4,), 2), ((2.6,), 3)]),
    ("pages_needed",                         # ceil division, reached for with //
     "Write pages_needed(books, per_page): how many shelf pages hold all the "
     "books at per_page books per page.",
     [((10, 3), 4), ((9, 3), 3), ((0, 3), 0), ((1, 3), 1), ((6, 3), 2)]),
    ("shelve_order",                         # composite sort key (ties by title)
     "Write shelve_order(books): given (title, rating) pairs, return the titles "
     "sorted by rating from high to low, breaking ties by title alphabetically.",
     [(([("b", 5), ("a", 5), ("c", 3)],), ["a", "b", "c"]),
      (([("x", 1), ("y", 2)],), ["y", "x"]), (([],), []),
      (([("c", 3), ("a", 3), ("b", 3)],), ["a", "b", "c"])]),
    ("percent_read",                         # zero denom + integer rounding
     "Write percent_read(read, total): the percent of books read as an integer "
     "0-100, rounded to nearest; if total is 0 return 0.",
     [((1, 4), 25), ((3, 4), 75), ((1, 3), 33), ((2, 3), 67), ((0, 0), 0),
      ((5, 5), 100)]),
    ("median_pages",                         # even-length averages two middles
     "Write median_pages(counts): return the median of the page counts; for an "
     "even count, average the two middle values.",
     [(([1, 2, 3, 4],), 2.5), (([3, 1, 2],), 2.0), (([5],), 5.0),
      (([4, 1, 7, 2],), 3.0)]),
    ("shelf_label",                          # singular vs plural
     "Write shelf_label(n): return '1 book' for one book, otherwise 'N books' "
     "(so 0 -> '0 books', 3 -> '3 books').",
     [((1,), "1 book"), ((0,), "0 books"), ((2,), "2 books"), ((11,), "11 books")]),
    ("reading_streak",                       # longest run of consecutive 1s
     "Write reading_streak(days): given a list of 1/0 flags for whether the "
     "reader read each day, return the length of the longest run of 1s.",
     [(([1, 1, 0, 1, 1, 1],), 3), (([0, 0, 0],), 0), (([1, 1, 1],), 3),
      (([],), 0), (([1, 0, 1],), 1)]),
    ("format_price",                         # cents need zero-padding to 2 places
     "Write format_price(cents): format a whole number of cents as a dollar "
     "string like '$12.34' (so 5 -> '$0.05', 100 -> '$1.00').",
     [((1234,), "$12.34"), ((5,), "$0.05"), ((100,), "$1.00"), ((0,), "$0.00"),
      ((99,), "$0.99")]),
    ("rounded_rating",                       # mean THEN round half up to a star
     "Write rounded_rating(scores): average the star scores and round the mean "
     "to the nearest whole star, rounding a half up.",
     [(([5, 4],), 5), (([3, 4, 4],), 4), (([2, 3],), 3), (([1, 1, 1],), 1),
      (([4, 5, 5, 5],), 5)]),
    # anchors from the live suite, for calibration
    ("nightly_pace",                         # empty list -> ZeroDivision (anchor)
     "Write nightly_pace(counts): return the mean of the page counts in the list.",
     [(([1, 2, 3],), 2.0), (([10],), 10.0), (([],), 0.0), (([2, 4],), 3.0)]),
    ("runner_up",                            # forgets DISTINCT (anchor)
     "Write runner_up(scores): return the second highest quiz score in scores.",
     [(([5, 5, 3],), 3), (([1, 2, 3, 4],), 3), (([10, 8, 10, 8],), 8), (([7, 3],), 3)]),
]


def screen(repeats, roster):
    P.temp = 0.7
    K = 4
    only = os.environ.get("REFLEX_TASKS")        # comma list to screen a subset
    tasks = ([t for t in CANDIDATES if t[0] in only.split(",")] if only
             else CANDIDATES)
    print(f"spread screen  K={K} repeats={repeats} models={len(roster)} "
          f"tasks={len(tasks)}\n", flush=True)
    for name, spec, cases in tasks:
        P.SUITE = [(name, spec, cases)]            # one-task suite for resolution
        print(f"== {name} ==", flush=True)
        for model in roster:
            blind = P.run_condition(False, model, K, 0.7, repeats)
            refl  = P.run_condition(True,  model, K, 0.7, repeats)
            lift = refl[-1] - blind[-1]
            print(f"   {model:<22} try1 {blind[0]:4.0%}  blind@{K} {blind[-1]:4.0%}"
                  f"  reflex@{K} {refl[-1]:4.0%}  (lift {lift:+.0%})", flush=True)
        print(flush=True)


if __name__ == "__main__":
    repeats = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    roster = sys.argv[2:] or ["qwen2.5-coder:latest", "mistral:latest",
                              "llama3.1:latest", "tinyllama:1.1b"]
    screen(repeats, roster)
