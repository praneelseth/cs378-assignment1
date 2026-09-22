"""Part 1 sanity check: the reference must pass against its own output, and
the harness must reject a deliberately wrong answer. Exits nonzero on failure.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "reference"))
from attention_ref import (attention_reference, check_against_reference,  # noqa: E402
                           make_inputs)


def main():
    fails = 0
    for causal in (False, True):
        Q, K, V = make_inputs(2, 4, 128, 64)
        ref = attention_reference(Q, K, V, causal=causal)
        # (a) reference vs itself: must PASS
        if not check_against_reference(ref, Q, K, V, causal=causal,
                                       label="self"):
            fails += 1
        # (b) a wrong answer (zeros): must FAIL -> so we invert the check
        wrong = np.zeros_like(ref)
        if check_against_reference(wrong, Q, K, V, causal=causal,
                                   label="wrong(expect FAIL)"):
            print("  harness accepted a wrong answer -- ASSIGN1_1_2 is broken")
            fails += 1
    print("ALL PASS" if fails == 0 else f"{fails} FAILURES")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
