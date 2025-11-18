import itertools
import os
import unittest

import numpy as np
from tqdm import tqdm

import ofc_cpp as _CPP


MASK64 = (1 << 64) - 1
SAMPLES_PER_ROUND = [48, 36, 24, 16, 8]
SHOW_PROGRESS = os.getenv("SFL_TEST_PROGRESS", "1") != "0"


def maybe_tqdm(iterable, desc):
    if SHOW_PROGRESS:
        return tqdm(iterable, desc=desc, leave=False)
    return iterable


def fnv_mix(seed, value):
    seed ^= (int(value) + 1) & MASK64
    return (seed * 1099511628211) & MASK64


def hash_state(board, round_idx, deck):
    h = 1469598103934665603
    h = fnv_mix(h, round_idx)
    for card in board:
        h = fnv_mix(h, card)
    for card in deck:
        h = fnv_mix(h, card)
    return h


def mix64(z):
    z &= MASK64
    z ^= (z >> 12)
    z &= MASK64
    z ^= (z << 25) & MASK64
    z ^= (z >> 27)
    z &= MASK64
    return (z * 2685821657736338717) & MASK64


def sample_cards(seed, sample_idx, deck, take):
    pool = list(deck)
    chosen = []
    state = (seed + 0x9E3779B97F4A7C15 * (sample_idx + 1)) & MASK64
    for i in range(take):
        if not pool:
            break
        state = mix64(state + i + 1)
        idx = int(state % len(pool))
        chosen.append(pool.pop(idx))
    return chosen


def split_rows(board):
    bottom = [c for c in board[:5] if c >= 0]
    middle = [c for c in board[5:10] if c >= 0]
    top = [c for c in board[10:13] if c >= 0]
    return bottom, middle, top


def merge_row(base, addition, size):
    merged = list(base) + list(addition)
    if len(merged) != size:
        raise AssertionError("Row merge size mismatch")
    return merged


def subtract_cards(cards, to_remove):
    remaining = list(cards)
    for card in to_remove:
        remaining.remove(card)
    return remaining


def combinations(cards, choose):
    if choose == 0:
        return [tuple()]
    if choose > len(cards):
        return []
    return list(itertools.combinations(cards, choose))


def score_completed(bottom, middle, top):
    bottom_arr = np.array(bottom, dtype=np.int16)
    middle_arr = np.array(middle, dtype=np.int16)
    top_arr = np.array(top, dtype=np.int16)
    score, _ = _CPP.score_board_from_ints(bottom_arr, middle_arr, top_arr)
    return float(score)


def rollout_value(board, round_idx, deck):
    bottom, middle, top = split_rows(board)
    bottom_need = 5 - len(bottom)
    middle_need = 5 - len(middle)
    top_need = 3 - len(top)
    total_need = bottom_need + middle_need + top_need

    if total_need == 0:
        return score_completed(bottom, middle, top)
    if len(deck) < total_need:
        return -6.0

    samples = max(1, SAMPLES_PER_ROUND[min(round_idx, 4)])
    seed = hash_state(board, round_idx, deck)
    total = 0.0

    iterator = maybe_tqdm(range(samples), desc=f"Samples (round {round_idx})")
    for sample_idx in iterator:
        sample = sample_cards(seed, sample_idx, deck, total_need)
        if len(sample) < total_need:
            total += -6.0
            continue

        best = float("-inf")
        bottom_variants = combinations(sample, bottom_need)
        if not bottom_variants:
            bottom_variants = [tuple()]
        for bottom_add in bottom_variants:
            after_bottom = subtract_cards(sample, bottom_add)
            middle_variants = combinations(after_bottom, middle_need)
            if not middle_variants:
                middle_variants = [tuple()]
            for middle_add in middle_variants:
                after_middle = subtract_cards(after_bottom, middle_add)
                if len(after_middle) != top_need:
                    continue
                bottom_full = merge_row(bottom, bottom_add, 5)
                middle_full = merge_row(middle, middle_add, 5)
                top_full = merge_row(top, after_middle, 3)
                score = score_completed(bottom_full, middle_full, top_full)
                if score > best:
                    best = score
        if best == float("-inf"):
            best = -6.0
        total += best

    return total / samples


def reference_sfl_choose_action(board_arr, round_idx, draw_arr, deck_arr):
    best_score = float("-inf")
    best_idx = -1
    board_list = board_arr.tolist()
    deck_list = deck_arr.tolist()

    if round_idx == 0:
        placements = _CPP.legal_actions_round0(board_arr)
        action_iter = maybe_tqdm(range(placements.shape[0]), desc="Round 0 actions")
        for idx in action_iter:
            slots = placements[idx, :, 1].astype(np.int16)
            next_board, next_round, _, next_deck, _ = _CPP.step_state_round0(
                board_arr, draw_arr, deck_arr, slots
            )
            score = rollout_value(next_board.tolist(), int(next_round), next_deck.tolist())
            if score > best_score:
                best_score = score
                best_idx = idx
    else:
        keeps, places = _CPP.legal_actions_rounds1to4(board_arr, int(round_idx))
        action_iter = maybe_tqdm(range(keeps.shape[0]), desc=f"Round {round_idx} actions")
        for idx in action_iter:
            k0, k1 = map(int, keeps[idx])
            p00, p01 = map(int, places[idx, 0])
            p10, p11 = map(int, places[idx, 1])
            next_board, next_round, _, next_deck, _ = _CPP.step_state(
                board_arr, int(round_idx), draw_arr, deck_arr,
                k0, k1, p00, p01, p10, p11
            )
            score = rollout_value(next_board.tolist(), int(next_round), next_deck.tolist())
            if score > best_score:
                best_score = score
                best_idx = idx

    return best_idx


def build_deck(exclude):
    excluded = set(c for c in exclude if c >= 0)
    return [c for c in range(52) if c not in excluded]


def round0_state():
    board = [-1] * 13
    draw = [0, 9, 18, 27, 36]
    deck = build_deck(draw)
    return board, 0, draw, deck


def round2_state():
    board = [
        0, 4, -1, -1, -1,
        12, 13, -1, -1, -1,
        30, -1, -1,
    ]
    draw = [21, 25, 29]
    exclude = [c for c in board if c >= 0] + draw
    deck = build_deck(exclude)
    return board, 2, draw, deck


def round4_state():
    board = [
        0, 8, 16, 24, 32,
        5, 13, 21, 28, -1,
        40, 44, -1,
    ]
    draw = [33, 34, 35]
    exclude = [c for c in board if c >= 0] + draw
    deck = build_deck(exclude)
    return board, 4, draw, deck


class TestSFLPolicy(unittest.TestCase):
    def _assert_matches_reference(self, builder):
        board, round_idx, draw, deck = builder()
        board_arr = np.array(board, dtype=np.int16)
        draw_arr = np.array(draw, dtype=np.int16)
        deck_arr = np.array(deck, dtype=np.int16)
        cpp_idx = _CPP.sfl_choose_action(board_arr, round_idx, draw_arr, deck_arr)
        ref_idx = reference_sfl_choose_action(board_arr, round_idx, draw_arr, deck_arr)
        self.assertEqual(cpp_idx, ref_idx)

    def test_round0_matches_reference(self):
        self._assert_matches_reference(round0_state)

    def test_round2_matches_reference(self):
        self._assert_matches_reference(round2_state)

    def test_round4_matches_reference(self):
        self._assert_matches_reference(round4_state)


if __name__ == "__main__":
    unittest.main()

