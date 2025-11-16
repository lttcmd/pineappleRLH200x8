import time
import numpy as np
import random
from ofc_types import Card, Rank, Suit
from ofc_scoring import score_board

try:
    import ofc_cpp as cpp
except Exception:
    cpp = None

def int_to_card(i: int) -> Card:
    rank = Rank((i // 4) + 2)
    suit = Suit(i % 4)
    return Card(rank, suit)

def bench_scoring(n=10000):
    deck = list(range(52))
    cases = []
    for _ in range(n):
        random.shuffle(deck)
        bottom = deck[:5]; middle = deck[5:10]; top = deck[10:13]
        cases.append((bottom[:], middle[:], top[:]))

    # Python
    t0 = time.time()
    s = 0.0
    for b, m, t in cases:
        sb, foul = score_board([int_to_card(i) for i in b],
                               [int_to_card(i) for i in m],
                               [int_to_card(i) for i in t])
        s += sb
    t1 = time.time()
    py_ms = (t1 - t0) * 1000.0

    print(f"Python scoring: {n} boards in {py_ms:.1f} ms ({n/(t1-t0):.0f} boards/s)")

    if cpp is not None:
        t0 = time.time()
        s = 0.0
        for b, m, t in cases:
            sb, foul = cpp.score_board_from_ints(np.array(b, dtype=np.int16),
                                                 np.array(m, dtype=np.int16),
                                                 np.array(t, dtype=np.int16))
            s += sb
        t1 = time.time()
        cpp_ms = (t1 - t0) * 1000.0
        print(f"C++ scoring:   {n} boards in {cpp_ms:.1f} ms ({n/(t1-t0):.0f} boards/s)")
    else:
        print("C++ module not available")

if __name__ == "__main__":
    bench_scoring()


