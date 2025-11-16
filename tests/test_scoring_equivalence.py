import random
import numpy as np
import os
import pytest

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

@pytest.mark.skipif(cpp is None, reason="C++ module not built")
def test_cpp_python_scoring_agree(num_samples: int = 100):
    deck = list(range(52))
    for _ in range(num_samples):
        random.shuffle(deck)
        bottom_ints = deck[:5]
        middle_ints = deck[5:10]
        top_ints = deck[10:13]
        bottom = [int_to_card(i) for i in bottom_ints]
        middle = [int_to_card(i) for i in middle_ints]
        top = [int_to_card(i) for i in top_ints]

        b = np.array(bottom_ints, dtype=np.int16)
        m = np.array(middle_ints, dtype=np.int16)
        t = np.array(top_ints, dtype=np.int16)

        s_cpp, foul_cpp = cpp.score_board_from_ints(b, m, t)
        s_py, foul_py = score_board(bottom, middle, top)

        # Allow exact match on foul flag, and score equality
        assert bool(foul_cpp) == bool(foul_py)
        assert abs(float(s_cpp) - float(s_py)) < 1e-5


