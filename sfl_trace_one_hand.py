"""
Trace a single full OFC hand using the native SFL heuristic, with verbose output.

Run (from project root, in PowerShell):
    python sfl_trace_one_hand.py --seed 42 --fast

This will:
  - Shuffle a 52-card deck
  - Deal 5 cards
  - Show all round-0 layouts and which one SFL chooses
  - For each of the next 4 rounds:
      * Deal 3 cards
      * Show all (keep, placement) actions
      * Highlight the action SFL chooses
  - Show the final board and its canonical OFC score.
"""

from __future__ import annotations

import argparse
import os
import random
from typing import List, Tuple

import numpy as np

import ofc_cpp
from sfl_interactive import (
    int_to_card,
    format_board,
    display_round0_actions,
    display_postflop_actions,
)


def deal_initial_hand(rng: random.Random) -> Tuple[List[int], List[int], List[int]]:
    """Shuffle a fresh deck and deal the initial 5 cards from the back (C++ style)."""
    deck = list(range(52))
    rng.shuffle(deck)
    initial5 = []
    for _ in range(5):
        initial5.append(deck.pop())
    return deck, initial5, list(initial5[:3])


def step_round0(
    board: List[int],
    deck: List[int],
    initial5: List[int],
) -> Tuple[List[int], List[int], List[int]]:
    """Run round 0 using SFL, with full action dump."""
    board_np = np.array(board, dtype=np.int16)
    draw_np = np.array(initial5, dtype=np.int16)
    deck_np = np.array(deck, dtype=np.int16)

    placements = ofc_cpp.legal_actions_round0(board_np)
    if placements.shape[0] == 0:
        print("No legal round-0 actions (board likely already filled?).")
        return board, deck, [-1, -1, -1]

    # Show all legal actions in a human-readable way.
    print("\n=== Round 0 ===")
    print("Initial 5:", ", ".join(int_to_card(c) for c in initial5))
    display_round0_actions(initial5, placements)

    best_idx = int(ofc_cpp.sfl_choose_action(board_np, 0, draw_np, deck_np))
    if best_idx < 0 or best_idx >= placements.shape[0]:
        best_idx = 0
    print(f"\nSFL chooses round-0 action: {best_idx}")

    # Apply chosen action via the C++ round-0 step helper.
    slots_np = np.full(5, -1, dtype=np.int16)
    for i in range(5):
        card_idx = int(placements[best_idx, i, 0])
        slot_idx = int(placements[best_idx, i, 1])
        if 0 <= card_idx < 5:
            slots_np[card_idx] = slot_idx

    next_board, next_round, next_draw, next_deck, done_flag = ofc_cpp.step_state_round0(
        board_np, draw_np, deck_np, slots_np
    )

    board_out = next_board.astype(np.int16).tolist()
    deck_out = next_deck.astype(np.int16).tolist()
    draw3_out = next_draw.astype(np.int16).tolist()

    print("\nBoard after round 0:")
    print(format_board(board_out))
    print("Next draw (round 1):", ", ".join(int_to_card(c) for c in draw3_out))
    print("Done flag:", bool(done_flag))

    return board_out, deck_out, draw3_out


def step_postflop_round(
    board: List[int],
    deck: List[int],
    draw3: List[int],
    round_idx: int,
) -> Tuple[List[int], List[int], List[int], bool]:
    """Run a postflop round (1..4) using SFL, with full action dump."""
    board_np = np.array(board, dtype=np.int16)
    draw_np = np.array(draw3, dtype=np.int16)
    deck_np = np.array(deck, dtype=np.int16)

    keeps, placements = ofc_cpp.legal_actions_rounds1to4(board_np, round_idx)
    if keeps.shape[0] == 0:
        print(f"\n=== Round {round_idx} ===")
        print("No legal actions (board probably full).")
        return board, deck, draw3, True

    print(f"\n=== Round {round_idx} ===")
    print("Current draw:", ", ".join(int_to_card(c) for c in draw3))
    display_postflop_actions(draw3, keeps, placements)

    best_idx = int(ofc_cpp.sfl_choose_action(board_np, round_idx, draw_np, deck_np))
    if best_idx < 0 or best_idx >= keeps.shape[0]:
        best_idx = 0
    print(f"\nSFL chooses round-{round_idx} action: {best_idx}")

    k = keeps[best_idx]
    p = placements[best_idx]
    keep_i, keep_j = int(k[0]), int(k[1])
    p00, p01 = int(p[0, 0]), int(p[0, 1])
    p10, p11 = int(p[1, 0]), int(p[1, 1])

    next_board, next_round, next_draw, next_deck, done_flag = ofc_cpp.step_state(
        board_np,
        int(round_idx),
        draw_np,
        deck_np,
        keep_i,
        keep_j,
        p00,
        p01,
        p10,
        p11,
    )

    board_out = next_board.astype(np.int16).tolist()
    deck_out = next_deck.astype(np.int16).tolist()
    draw3_out = next_draw.astype(np.int16).tolist()

    print("\nBoard after this round:")
    print(format_board(board_out))
    if round_idx < 4:
        print(
            f"Next draw (round {round_idx + 1}):",
            ", ".join(int_to_card(c) for c in draw3_out),
        )
    print("Done flag:", bool(done_flag))

    return board_out, deck_out, draw3_out, bool(done_flag)


def score_final_board(board: List[int]) -> None:
    """Score the completed board using canonical C++ scoring."""
    bottom = np.array(board[:5], dtype=np.int16)
    middle = np.array(board[5:10], dtype=np.int16)
    top = np.array(board[10:13], dtype=np.int16)
    score, fouled = ofc_cpp.score_board_from_ints(bottom, middle, top)

    print("\n=== Final Result ===")
    print("Final board:")
    print(format_board(board))
    if fouled:
        print("Result: FOUL")
    else:
        print(f"Result: score = {score:.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trace a single OFC hand played by the native SFL heuristic.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Random seed for deck shuffling.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use fast SFL rollouts (OFC_SFL_FAST=1) for quicker traces.",
    )
    parser.add_argument(
        "--hands",
        type=int,
        default=1,
        help="Number of hands to trace.",
    )
    args = parser.parse_args()

    if args.fast:
        os.environ["OFC_SFL_FAST"] = "1"
    else:
        os.environ.pop("OFC_SFL_FAST", None)

    rng = random.Random(args.seed)

    for hand_idx in range(args.hands):
        print("\n" + "=" * 80)
        print(f"Hand {hand_idx + 1} / {args.hands}")

        # Start from an empty board and a fresh deck for each hand.
        board = [-1] * 13
        deck, initial5, draw3 = deal_initial_hand(rng)

        print("Shuffled deck (top shown last 10 cards):")
        print(", ".join(int_to_card(c) for c in deck[-10:]))
        print("Initial 5 cards:", ", ".join(int_to_card(c) for c in initial5))

        # Round 0
        board, deck, draw3 = step_round0(board, deck, initial5)

        # Rounds 1..4
        done = False
        round_idx = 1
        while round_idx <= 4 and not done:
            board, deck, draw3, done = step_postflop_round(
                board, deck, draw3, round_idx
            )
            round_idx += 1

        score_final_board(board)


if __name__ == "__main__":
    main()


