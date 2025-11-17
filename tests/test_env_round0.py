import copy
import random

import pytest

import ofc_env
from ofc_env import OfcEnv, State


def clone_state(state: State) -> State:
    return State(
        board=list(state.board),
        round=state.round,
        current_draw=list(state.current_draw),
        deck=list(state.deck),
        cards_placed_this_round=state.cards_placed_this_round,
    )


@pytest.mark.skipif(ofc_env._CPP is None, reason="C++ backend not available")
def test_round0_cpp_matches_python_step():
    random.seed(1234)
    py_env = OfcEnv(soft_mask=False, use_cpp=False)
    cpp_env = OfcEnv(soft_mask=False, use_cpp=True)
    initial_state = py_env.reset()
    cpp_actions = cpp_env.legal_actions(initial_state)
    assert cpp_actions, "Expected at least one action from C++ generator"
    action = cpp_actions[0]

    cpp_state, _, _ = cpp_env.step(clone_state(initial_state), action)
    py_state, _, _ = py_env.step(clone_state(initial_state), action)

    assert cpp_state.board == py_state.board
    assert cpp_state.round == py_state.round == 1
    assert cpp_state.current_draw == py_state.current_draw
    assert sum(1 for card in cpp_state.board if card is not None) == 5

