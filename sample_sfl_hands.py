import argparse
import random
from typing import List, Tuple

import numpy as np

import ofc_cpp
from sfl_interactive import int_to_card, format_board


def reset_state(rng: random.Random) -> Tuple[List[int], List[int], List[int], int, bool]:
    """
    Python replica of the C++ SimpleState::reset_state logic used in sfl_policy.cpp.

    Returns:
        board: current 13-slot board (ints, -1 for empty)
        deck: remaining deck (list of 0..51)
        initial5: initial 5-card draw
        round_idx: current round index (0 at start)
        done: whether the episode is finished (False at reset)
    """
    deck = list(range(52))
    rng.shuffle(deck)
    board = [-1] * 13
    initial5 = []
    for _ in range(5):
        # Match C++: take from the back of the shuffled deck.
        initial5.append(deck.pop())
    draw3 = initial5[:3]
    round_idx = 0
    done = False
    return board, deck, initial5, draw3, round_idx, done


def play_hand_with_sfl(
    rng: random.Random,
) -> Tuple[List[int], List[int], float, bool]:
    """
    Simulate a single full hand using the native SFL heuristic via ofc_cpp.

    Returns:
        initial5: the initial 5 cards dealt
        final_board: 13-card board (bottom[5], middle[5], top[3])
        score: OFC score from cpp (royalties – foul penalty)
        fouled: True if the board fouled according to cpp scoring
    """
    board, deck, initial5, draw3, round_idx, done = reset_state(rng)

    while not done:
        board_np = np.array(board, dtype=np.int16)
        deck_np = np.array(deck, dtype=np.int16)

        if round_idx == 0:
            draw_np = np.array(initial5, dtype=np.int16)
            placements = ofc_cpp.legal_actions_round0(board_np)
            action_count = int(placements.shape[0])
            if action_count <= 0:
                # No legal placements, treat as terminal.
                break

            action_idx = int(
                ofc_cpp.sfl_choose_action(board_np, round_idx, draw_np, deck_np)
            )
            if action_idx < 0 or action_idx >= action_count:
                action_idx = 0

            # Build slots array like C++: length-5, -1 default, index by card_idx.
            slots_np = np.full(5, -1, dtype=np.int16)
            for i in range(5):
                card_idx = int(placements[action_idx, i, 0])
                slot_idx = int(placements[action_idx, i, 1])
                if 0 <= card_idx < 5:
                    slots_np[card_idx] = slot_idx
            print("Round0 placements row:", placements[action_idx])
            print("Round0 slots_np:", slots_np)

            next_board, next_round, next_draw, next_deck, done_flag = (
                ofc_cpp.step_state_round0(board_np, draw_np, deck_np, slots_np)
            )
            print("Round0 next_board ints:", next_board.astype(int).tolist())
        else:
            keeps, places = ofc_cpp.legal_actions_rounds1to4(board_np, round_idx)
            action_count = int(keeps.shape[0])
            if action_count <= 0:
                break

            draw_np = np.array(draw3, dtype=np.int16)
            action_idx = int(
                ofc_cpp.sfl_choose_action(board_np, round_idx, draw_np, deck_np)
            )
            if action_idx < 0 or action_idx >= action_count:
                action_idx = 0

            k = keeps[action_idx]
            p = places[action_idx]
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

        board = next_board.astype(np.int16).tolist()
        round_idx = int(next_round)
        draw3 = next_draw.astype(np.int16).tolist()
        deck = next_deck.astype(np.int16).tolist()
        done = bool(done_flag)
        print(f"After round {round_idx}, board ints:", board)

    bottom = np.array(board[:5], dtype=np.int16)
    middle = np.array(board[5:10], dtype=np.int16)
    top = np.array(board[10:13], dtype=np.int16)
    score, fouled = ofc_cpp.score_board_from_ints(bottom, middle, top)

    return initial5, board, float(score), bool(fouled)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample full OFC hands played by the native SFL heuristic."
    )
    parser.add_argument(
        "--hands",
        type=int,
        default=10,
        help="Number of hands to generate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Random seed for shuffling decks.",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)

    for idx in range(args.hands):
        initial5, board, score, fouled = play_hand_with_sfl(rng)
        print("=" * 60)
        print(f"Hand {idx + 1}")
        print("Initial 5:", ", ".join(int_to_card(c) for c in initial5))
        print("Final board:")
        print(format_board(board))
        # Debug: also show raw integer board for inspection.
        print("Board ints:", board)
        if fouled:
            print("Result: FOUL")
        else:
            print(f"Result: score = {score:.1f}")


if __name__ == "__main__":
    main()



