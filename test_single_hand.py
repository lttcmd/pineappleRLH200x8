"""
Test script: Play a single hand of OFC following the exact pattern:
1. Place initial 5 cards
2. Deal 3 cards, place 2 discard 1
3. Deal 3 cards, place 2 discard 1
4. Deal 3 cards, place 2 discard 1
5. Deal 3 cards, place 2 discard 1
"""
import os
# Disable C++ to test Python path
os.environ['OFC_USE_CPP'] = '0'

from ofc_env import OfcEnv, State, Action
from ofc_scoring import score_board, validate_board, total_royalties
import random


def format_card(card):
    """Format a card for display."""
    if card is None:
        return "  "
    rank_str = str(card.rank.value) if card.rank.value <= 9 else card.rank.name[0]
    suit_str = card.suit.name[0]
    return f"{rank_str}{suit_str}"


def print_board(state):
    """Print the current board state."""
    bottom = [state.board[i] for i in range(5)]
    middle = [state.board[i] for i in range(5, 10)]
    top = [state.board[i] for i in range(10, 13)]
    
    print("\n" + "="*50)
    print("Current Board:")
    print(f"  Top (3):    {' '.join(format_card(c) for c in top)}")
    print(f"  Middle (5): {' '.join(format_card(c) for c in middle)}")
    print(f"  Bottom (5): {' '.join(format_card(c) for c in bottom)}")
    print(f"  Round: {state.round}")
    print(f"  Cards in deck: {len(state.deck)}")
    print("="*50)


def test_single_hand():
    """Play one complete hand following the exact pattern."""
    env = OfcEnv()
    state = env.reset()
    
    print("\n" + "="*50)
    print("TEST: Single Hand Play")
    print("="*50)
    
    # Step 1: Place initial 5 cards (Round 0)
    print("\n[STEP 1] Round 0: Place initial 5 cards")
    print(f"  Initial cards: {' '.join(format_card(c) for c in state.current_draw)}")
    
    legal_actions = env.legal_actions(state)
    if not legal_actions:
        print("ERROR: No legal actions for initial 5 cards!")
        return
    
    # Pick first legal action (places all 5 cards)
    action = legal_actions[0]
    print(f"  Action: keep_indices={action.keep_indices}, placements={action.placements}")
    print(f"  Board before: {[format_card(c) for c in state.board]}")
    
    state, reward, done = env.step(state, action)
    
    print(f"  Board after: {[format_card(c) for c in state.board]}")
    print_board(state)
    print(f"  Cards placed: {sum(1 for c in state.board if c is not None)}/13")
    
    if done:
        print("  Board completed early!")
        final_score = env.score(state)
        print(f"  Final score: {final_score}")
        return
    
    # Steps 2-5: Deal 3 cards, place 2 discard 1 (Rounds 1-4)
    for round_num in range(1, 5):
        print(f"\n[STEP {round_num + 1}] Round {round_num}: Deal 3 cards, place 2 discard 1")
        
        if not state.current_draw:
            print("  ERROR: No cards in current_draw!")
            break
        
        print(f"  Cards dealt: {' '.join(format_card(c) for c in state.current_draw)}")
        
        legal_actions = env.legal_actions(state)
        if not legal_actions:
            print("  ERROR: No legal actions!")
            break
        
        # Pick first legal action (keeps 2, discards 1, places 2)
        action = legal_actions[0]
        kept_cards = [state.current_draw[i] for i in action.keep_indices]
        discarded = [c for i, c in enumerate(state.current_draw) if i not in action.keep_indices]
        
        print(f"  Action: keep_indices={action.keep_indices}, placements={action.placements}")
        print(f"  Keeping: {' '.join(format_card(c) for c in kept_cards)}")
        print(f"  Discarding: {' '.join(format_card(c) for c in discarded)}")
        print(f"  Board before: {sum(1 for c in state.board if c is not None)}/13 cards")
        
        state, reward, done = env.step(state, action)
        
        print(f"  Board after: {sum(1 for c in state.board if c is not None)}/13 cards")
        
        print_board(state)
        print(f"  Cards placed: {sum(1 for c in state.board if c is not None)}/13")
        
        if done:
            print(f"  Board completed at round {round_num}!")
            break
    
    # Final scoring
    print("\n" + "="*50)
    print("FINAL RESULT")
    print("="*50)
    
    final_score = env.score(state)
    bottom = [state.board[i] for i in range(5)]
    middle = [state.board[i] for i in range(5, 10)]
    top = [state.board[i] for i in range(10, 13)]
    
    is_complete = all(slot is not None for slot in state.board)
    
    # Filter out None cards before validation
    bottom_clean = [c for c in bottom if c is not None]
    middle_clean = [c for c in middle if c is not None]
    top_clean = [c for c in top if c is not None]
    
    if not is_complete:
        print(f"  WARNING: Board is incomplete! Only {sum(1 for c in state.board if c is not None)}/13 cards placed")
        print(f"  Bottom: {len(bottom_clean)}/5, Middle: {len(middle_clean)}/5, Top: {len(top_clean)}/3")
        return final_score
    
    is_valid, reason = validate_board(bottom, middle, top)
    royalties = total_royalties(bottom, middle, top) if is_valid else 0
    
    print(f"  Board complete: {is_complete}")
    print(f"  Board valid: {is_valid}")
    if not is_valid:
        print(f"  Foul reason: {reason}")
    print(f"  Final score: {final_score:.1f}")
    if is_valid:
        print(f"  Royalties: {royalties}")
    else:
        print(f"  Penalty: -6 points")
    print("="*50 + "\n")
    
    return final_score


if __name__ == '__main__':
    test_single_hand()

