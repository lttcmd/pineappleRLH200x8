"""
Interactive CLI tool to inspect SFL's choices round-by-round.

Usage:
    python sfl_interactive.py

You will be prompted to enter the initial 5 cards (e.g. "Ah,Kd,Qc,9s,2d").
After each round you can supply the next draw of 3 cards (or press Enter to
auto-draw from the remaining unseen cards). The script prints every legal
action, highlights the SFL suggestion, and lets you override the choice.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

import ofc_cpp

RANKS = "23456789TJQKA"
SUITS = "cdhs"


def card_to_int(card: str) -> int:
    card = card.strip().lower()
    if len(card) != 2 or card[0].upper() not in RANKS or card[1] not in SUITS:
        raise ValueError(f"Invalid card: {card}")
    rank_idx = RANKS.index(card[0].upper())
    suit_idx = SUITS.index(card[1])
    return rank_idx * 4 + suit_idx


def int_to_card(card_int: int) -> str:
    if card_int < 0:
        return "--"
    rank_idx = card_int // 4
    suit_idx = card_int % 4
    return f"{RANKS[rank_idx]}{SUITS[suit_idx]}"


def slot_label(slot: int) -> str:
    if slot < 5:
        return f"B{slot}"
    if slot < 10:
        return f"M{slot - 5}"
    return f"T{slot - 10}"


def format_board(board: Sequence[int]) -> str:
    bottom = " ".join(int_to_card(c) for c in board[:5])
    middle = " ".join(int_to_card(c) for c in board[5:10])
    top = " ".join(int_to_card(c) for c in board[10:13])
    return f"Bottom: {bottom}\nMiddle: {middle}\nTop   : {top}"


def prompt_cards(prompt: str, expected: int, used: set[int]) -> List[int]:
    while True:
        raw = input(prompt).strip()
        if not raw:
            # Auto-draw lowest remaining cards
            remaining = sorted(set(range(52)) - used)
            if len(remaining) < expected:
                raise ValueError("Not enough remaining cards to auto-draw.")
            cards = remaining[:expected]
            print("Auto draw:", ", ".join(int_to_card(c) for c in cards))
        else:
            tokens = [t.strip() for t in raw.split(",") if t.strip()]
            if len(tokens) != expected:
                print(f"Please enter exactly {expected} cards.")
                continue
            try:
                cards = [card_to_int(tok) for tok in tokens]
            except ValueError as exc:
                print(exc)
                continue
            if len(set(cards)) != len(cards):
                print("Duplicate cards in draw.")
                continue
            if any(c in used for c in cards):
                print("One or more cards already used.")
                continue
        used.update(cards)
        return cards


def display_round0_actions(draw: List[int], placements: np.ndarray) -> None:
    print(f"\nRound 0 actions ({placements.shape[0]} total):")
    for idx, placement in enumerate(placements):
        pairs = [
            f"{int_to_card(draw[int(card_idx)])}->{slot_label(int(slot_idx))}"
            for card_idx, slot_idx in placement
        ]
        print(f"  Action {idx:02d}: " + ", ".join(pairs))


def display_postflop_actions(
    draw: List[int], keeps: np.ndarray, placements: np.ndarray
) -> None:
    print(f"\nActions ({placements.shape[0]} total):")
    for idx in range(placements.shape[0]):
        keep_pair = keeps[idx]
        kept_cards = [draw[int(keep_pair[0])], draw[int(keep_pair[1])]]
        keep_desc = f"keep [{int_to_card(kept_cards[0])}, {int_to_card(kept_cards[1])}]"
        moves = []
        for move in placements[idx]:
            kept_slot = int(move[0])
            slot_idx = int(move[1])
            moves.append(f"{int_to_card(kept_cards[kept_slot])}->{slot_label(slot_idx)}")
        print(f"  Action {idx:02d}: {keep_desc}; placements: " + ", ".join(moves))


def choose_action(default_idx: int, max_idx: int) -> int:
    while True:
        raw = input(f"Choose action [default {default_idx}]: ").strip()
        if not raw:
            return default_idx
        if not raw.isdigit():
            print("Enter a numeric index.")
            continue
        val = int(raw)
        if 0 <= val < max_idx:
            return val
        print("Index out of range.")


def apply_round0_action(
    board: List[int], draw: List[int], placement: np.ndarray
) -> None:
    for card_idx, slot_idx in placement:
        board[int(slot_idx)] = draw[int(card_idx)]


def apply_postflop_action(
    board: List[int],
    draw: List[int],
    keeps: np.ndarray,
    placements: np.ndarray,
    action_idx: int,
) -> None:
    keep_pair = keeps[action_idx]
    kept_cards = [draw[int(keep_pair[0])], draw[int(keep_pair[1])]]
    for move in placements[action_idx]:
        kept_ref = int(move[0])
        slot_idx = int(move[1])
        board[slot_idx] = kept_cards[kept_ref]


def run_round0(board: List[int], draw: List[int], remaining: set[int]) -> None:
    board_np = np.array(board, dtype=np.int16)
    draw_np = np.array(draw, dtype=np.int16)
    deck_np = np.array(sorted(remaining), dtype=np.int16)

    placements = ofc_cpp.legal_actions_round0(board_np)
    display_round0_actions(draw, placements)
    best_idx = ofc_cpp.sfl_choose_action(board_np, 0, draw_np, deck_np)
    print("SFL suggests action:", best_idx)
    choice = choose_action(best_idx, placements.shape[0])
    apply_round0_action(board, draw, placements[choice])


def run_postflop_round(
    board: List[int],
    draw: List[int],
    round_idx: int,
    remaining: set[int],
) -> None:
    board_np = np.array(board, dtype=np.int16)
    draw_np = np.array(draw, dtype=np.int16)
    deck_np = np.array(sorted(remaining), dtype=np.int16)

    keeps, placements = ofc_cpp.legal_actions_rounds1to4(board_np, round_idx)
    if keeps.shape[0] == 0:
        print("No legal actions available (board likely full).")
        return
    display_postflop_actions(draw, keeps, placements)
    best_idx = ofc_cpp.sfl_choose_action(board_np, round_idx, draw_np, deck_np)
    print("SFL suggests action:", best_idx)
    choice = choose_action(best_idx, keeps.shape[0])
    apply_postflop_action(board, draw, keeps, placements, choice)
    kept_indices = keeps[choice]
    kept_cards = [draw[int(kept_indices[0])], draw[int(kept_indices[1])]]
    discarded = [c for c in draw if c not in kept_cards]
    if discarded:
        print("Discarded card:", int_to_card(discarded[0]))


def score_final_board(board: List[int]) -> None:
    bottom = np.array(board[:5], dtype=np.int16)
    middle = np.array(board[5:10], dtype=np.int16)
    top = np.array(board[10:], dtype=np.int16)
    score, fouled = ofc_cpp.score_board_from_ints(bottom, middle, top)
    print("\nFinal board:")
    print(format_board(board))
    if fouled:
        print("Result: FOUL (-6)")
    else:
        print(f"Result: {score:.1f} points")


def main() -> None:
    board = [-1] * 13
    used_cards: set[int] = set()

    print("Enter initial 5 cards (e.g. Ah,Kd,Qc,9s,2d). Leave blank for auto-draw.")
    initial_draw = prompt_cards("Initial 5 cards: ", 5, used_cards)
    remaining = set(range(52)) - used_cards
    current_draw = initial_draw

    for round_idx in range(5):
        print("\n" + "=" * 60)
        print(f"Round {round_idx} board:")
        print(format_board(board))
        print("Current draw:", ", ".join(int_to_card(c) for c in current_draw))

        if round_idx == 0:
            run_round0(board, current_draw, remaining)
        else:
            run_postflop_round(board, current_draw, round_idx, remaining)

        if round_idx == 4:
            break

        prompt = (
            f"Enter next 3 cards for round {round_idx + 1} "
            "(blank for auto-draw): "
        )
        current_draw = prompt_cards(prompt, 3, used_cards)
        remaining -= set(current_draw)

    score_final_board(board)


if __name__ == "__main__":
    main()

