"""Reproducer for the agentic.ipynb Reflexion trace demo (cell age-reflexion-trace).

Drives the REAL reflexion loop from genai.agent over a set of hidden-gotcha
coding tasks and logs each trajectory attempt by attempt, so we can pick a task
where the configured model (REFLEXION_MODEL = mistral:latest) GENUINELY fails on
the first try, then fixes it after reading its own one-sentence lesson. The demo
bakes the two real attempts it finds here; this script is what regenerates them.

Run: .venv/bin/python scripts/_reflexion_probe.py [model] [repeats]
"""
import sys
sys.path.insert(0, "chapters")
from genai.agent import reflexion, code_test, REFLEXION_MODEL, _crux

# Candidate hidden-gotcha tasks: terse spec, the test cases hold the real
# contract (the edge case a first attempt rarely anticipates). The comment names
# the gotcha and the stable wrong prior it provokes.
CANDIDATES = [
    ("average",                                          # empty -> ZeroDivision
     "Write average(nums): return the mean of the numbers in the list.",
     [(([1, 2, 3],), 2.0), (([10],), 10.0), (([],), 0.0), (([2, 4],), 3.0)]),
    ("dedupe",                                           # set() loses order
     "Write dedupe(xs): remove the duplicate items from the list xs.",
     [(([3, 1, 3, 2, 1],), [3, 1, 2]), (([1, 2, 3],), [1, 2, 3]),
      (([],), []), (([5, 5, 5],), [5])]),
    ("count_words",                                      # split() vs split(' ')
     "Write count_words(s): count the number of words in the string s.",
     [(("hello world",), 2), (("",), 0), (("   ",), 0), (("  a  b c ",), 3)]),
    ("second_largest",                                   # forgets DISTINCT
     "Write second_largest(xs): return the second largest number in the list xs.",
     [(([5, 5, 3],), 3), (([1, 2, 3, 4],), 3), (([10, 8, 10, 8],), 8),
      (([7, 3],), 3)]),
    ("rotate",                                           # k > len, k=0 wrap
     "Write rotate(xs, k): rotate the list xs to the right by k places.",
     [(([1, 2, 3, 4], 1), [4, 1, 2, 3]), (([1, 2, 3], 0), [1, 2, 3]),
      (([1, 2, 3], 4), [3, 1, 2]), (([], 2), [])]),
    ("median",                                           # even-length average
     "Write median(nums): return the median of the numbers in the list.",
     [(([3, 1, 2],), 2.0), (([1, 2, 3, 4],), 2.5), (([5],), 5.0),
      (([4, 1, 7, 2],), 3.0)]),
]


def run(model, repeats):
    print(f"model={model}  repeats={repeats}  max_trials=2", flush=True)
    for name, spec, cases in CANDIDATES:
        check = code_test(name, cases)
        first_fail = second_fix = 0
        sample = None
        for r in range(repeats):
            trace = reflexion(spec, check, model=model, max_trials=2)
            t1_ok = trace[0][1]
            t2_ok = trace[1][1] if len(trace) > 1 else t1_ok
            if not t1_ok:
                first_fail += 1
                if t2_ok:
                    second_fix += 1
                    if sample is None:           # keep one real fail->fix pair
                        sample = (trace[0], trace[1])
            tag = ("PASS@1" if t1_ok else
                   ("FAIL@1->PASS@2" if t2_ok else "FAIL@1->FAIL@2"))
            print(f"  [{name:14} r{r}] {tag}  | try1: {trace[0][2]}", flush=True)
        print(f"  >> {name}: failed try1 {first_fail}/{repeats}, "
              f"of those fixed by try2 {second_fix}/{first_fail or 1}", flush=True)
        if sample:
            (c1, _, d1, lesson), (c2, ok2, d2, _) = sample
            print("     --- real fail->fix sample ---")
            print("     TRY1:", _crux(c1).replace("\n", " | "))
            print("     TEST:", d1)
            print("     LESSON:", lesson)
            print("     TRY2:", _crux(c2).replace("\n", " | "))
            print("     TEST:", d2, "OK" if ok2 else "FAIL")
        print(flush=True)


if __name__ == "__main__":
    model   = sys.argv[1] if len(sys.argv) > 1 else REFLEXION_MODEL
    repeats = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    run(model, repeats)
