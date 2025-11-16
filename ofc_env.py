"""
Open Face Chinese Poker (OFC) Environment
Implements the core game rules and state management.
"""
import random
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from itertools import permutations, combinations
from ofc_types import Card, Rank, Suit
from ofc_scoring import validate_board, score_board
import os
_USE_CPP = os.getenv("OFC_USE_CPP", "1") == "1"
_CPP = None
if _USE_CPP:
    try:
        import ofc_cpp as _CPP
    except Exception:
        _CPP = None


# Heuristic caps to reduce CPU branching during action generation
MAX_INITIAL_PERMUTATIONS = 12           # cap for round-0 permutations (was 24)
MAX_EMPTY_SLOTS_CONSIDERED = 8          # consider at most this many empty slots per decision
MAX_SLOT_COMBOS_PER_PAIR = 6            # cap for (slot_i, slot_j) pairs per kept card pair (was 15)


@dataclass
class State:
    """Represents the current game state."""
    # Board: 13 slots total
    # bottom: slots 0-4 (5 cards)
    # middle: slots 5-9 (5 cards)
    # top: slots 10-12 (3 cards)
    board: List[Optional[Card]]  # 13 slots, None if empty
    
    # Current round/street (0-4)
    # Round 0: deal 5 cards (initial)
    # Rounds 1-4: deal 3 cards each (pineapple style)
    round: int
    
    # Current 3-card draw (only valid during rounds 1-4)
    current_draw: List[Card]  # 3 cards
    
    # Deck state (remaining cards)
    deck: List[Card]
    
    # Cards placed this round (for tracking)
    cards_placed_this_round: int


class Action:
    """Represents an action: which 2 cards to keep and where to place them."""
    def __init__(self, keep_indices: Tuple[int, int], placements: List[Tuple[int, int]]):
        """
        Args:
            keep_indices: Which 2 of the 3 cards to keep (e.g., (0, 1))
            placements: List of (card_index_in_keep, slot_index) pairs
        """
        self.keep_indices = keep_indices
        self.placements = placements  # [(0, slot), (1, slot)]
    
    def __repr__(self):
        return f"Action(keep={self.keep_indices}, place={self.placements})"


class OfcEnv:
    """Open Face Chinese Poker Environment."""
    
    def __init__(self):
        self.reset()
    
    def _create_deck(self) -> List[Card]:
        """Create a standard 52-card deck."""
        deck = []
        for rank in Rank:
            for suit in Suit:
                deck.append(Card(rank, suit))
        return deck
    
    def reset(self) -> State:
        """Reset environment and return initial state."""
        deck = self._create_deck()
        random.shuffle(deck)
        
        # Deal initial 5 cards
        initial_cards = [deck.pop() for _ in range(5)]
        
        # Create empty board
        board = [None] * 13
        
        state = State(
            board=board,
            round=0,
            current_draw=initial_cards,
            deck=deck,
            cards_placed_this_round=0
        )
        
        return state
    
    def legal_actions(self, state: State) -> List[Action]:
        """
        Get all legal actions from current state.
        Optimized version with minimal combinations.
        In round 0: must place all 5 cards
        In rounds 1-4: choose 2 of 3 cards, place them
        """
        # C++ accelerated path for rounds 1..4 to reduce Python overhead
        if _CPP is not None and state.round > 0:
            import numpy as _np
            board_arr = _np.array([c.to_int() if c is not None else -1 for c in state.board], dtype=_np.int16)
            keeps, places = _CPP.legal_actions_rounds1to4(board_arr, int(state.round))
            legal: List[Action] = []
            # Convert arrays to Action objects
            for i in range(keeps.shape[0]):
                k0 = int(keeps[i,0]); k1 = int(keeps[i,1])
                p00 = int(places[i,0,0]); p01 = int(places[i,0,1])
                p10 = int(places[i,1,0]); p11 = int(places[i,1,1])
                legal.append(Action(keep_indices=(k0, k1), placements=[(p00, p01), (p10, p11)]))
            return legal
        
        legal = []
        
        if state.round == 0:
            if _CPP is not None:
                import numpy as _np
                board_arr = _np.array([c.to_int() if c is not None else -1 for c in state.board], dtype=_np.int16)
                placements = _CPP.legal_actions_round0(board_arr)
                legal: List[Action] = []
                for i in range(placements.shape[0]):
                    placement5 = [(int(placements[i,j,0]), int(placements[i,j,1])) for j in range(5)]
                    legal.append(Action(keep_indices=tuple(range(5)), placements=placement5))
                return legal
            # Fallback Python path
            legal = []
            empty_slots = [i for i in range(13) if state.board[i] is None]
            if len(empty_slots) < 5:
                return []
            if len(empty_slots) > MAX_EMPTY_SLOTS_CONSIDERED:
                import random as _rand
                empty_slots = _rand.sample(empty_slots, MAX_EMPTY_SLOTS_CONSIDERED)
                if len(empty_slots) < 5:
                    empty_slots = [i for i in range(13) if state.board[i] is None]
            count = 0
            for perm in permutations(range(5)):
                placements = [(card_idx, empty_slots[i]) for i, card_idx in enumerate(perm)]
                legal.append(Action(keep_indices=tuple(range(5)), placements=placements))
                count += 1
                if count >= MAX_INITIAL_PERMUTATIONS:
                    break
        else:
            # Rounds 1-4: choose 2 of 3 cards
            empty_slots = [i for i in range(13) if state.board[i] is None]
            
            if len(empty_slots) < 2:
                return []  # Board is full
            
            # Subsample empty slots considered to keep combinations small
            if len(empty_slots) > MAX_EMPTY_SLOTS_CONSIDERED:
                import random as _rand
                empty_slots = _rand.sample(empty_slots, MAX_EMPTY_SLOTS_CONSIDERED)
            
            # Choose 2 of 3 cards (3 combinations: (0,1), (0,2), (1,2))
            for keep in combinations(range(3), 2):
                # For each pair of cards, try placing them in empty slots
                # Iterate combinations without materializing full list; cap per pair
                pair_count = 0
                for slot_pair in combinations(empty_slots, 2):
                    # Add both orderings to preserve some symmetry
                    legal.append(Action(keep_indices=keep, placements=[(0, slot_pair[0]), (1, slot_pair[1])]))
                    legal.append(Action(keep_indices=keep, placements=[(1, slot_pair[0]), (0, slot_pair[1])]))
                    pair_count += 1
                    if pair_count >= MAX_SLOT_COMBOS_PER_PAIR:
                        break
        
        return legal
    
    def _is_valid_placement(self, state: State, placements: List[Tuple[int, int]], 
                           keep_indices: Optional[Tuple[int, int]] = None) -> bool:
        """
        Check if a placement is valid.
        During play: Only checks basic slot validity (no constraint checking)
        Validation of bottom > middle > top happens only at final scoring.
        """
        # Basic slot validation only
        for card_idx, slot_idx in placements:
            if slot_idx < 0 or slot_idx >= 13:
                return False
            if state.board[slot_idx] is not None:
                return False
        
        # During play, allow any placement that fits in empty slots
        # Full validation (bottom > middle > top) only happens at scoring time
        # This allows games to complete even if the final board will be fouled
        # The model learns from both fouled and non-fouled outcomes
        return True
    
    def step(self, state: State, action: Action) -> Tuple[State, float, bool]:
        """
        Apply action and return (next_state, reward, done).
        Optimized with minimal copying.
        Reward is 0 during play, final score computed at end.
        """
        # C++ accelerated path for rounds 1..4
        if _CPP is not None:
            import numpy as _np
            # Prepare arrays for C++
            board_arr = _np.array([c.to_int() if c is not None else -1 for c in state.board], dtype=_np.int16)
            deck_arr = _np.array([c.to_int() for c in state.deck], dtype=_np.int16)
            if state.round == 0:
                # Round 0 specialized path
                current5 = _np.array([c.to_int() for c in state.current_draw], dtype=_np.int16)
                slots5 = _np.array([p[1] for p in action.placements], dtype=_np.int16)
                (b2, new_round, d2, deck2, done) = _CPP.step_state_round0(
                    board_arr, current5, deck_arr, slots5
                )
            else:
                draw_arr = _np.full((3,), -1, dtype=_np.int16)
                for i in range(min(3, len(state.current_draw))):
                    draw_arr[i] = state.current_draw[i].to_int()
                keep_i, keep_j = action.keep_indices
                (b2, new_round, d2, deck2, done) = _CPP.step_state(
                    board_arr, int(state.round), draw_arr, deck_arr,
                    int(keep_i), int(keep_j),
                    int(action.placements[0][0]), int(action.placements[0][1]),
                    int(action.placements[1][0]), int(action.placements[1][1])
                )
            # Convert back to Python State
            from ofc_types import Card, Rank, Suit
            def int_to_card(v: int):
                rank = Rank((v // 4) + 2)
                suit = Suit(v % 4)
                return Card(rank, suit)
            new_board = [int_to_card(int(v)) if int(v) >= 0 else None for v in b2.tolist()]
            next_draw = [int_to_card(int(v)) for v in d2.tolist() if int(v) >= 0]
            new_deck = [int_to_card(int(v)) for v in deck2.tolist()]
            new_state = State(
                board=new_board,
                round=new_round,
                current_draw=next_draw,
                deck=new_deck,
                cards_placed_this_round=len(action.placements)
            )
            return new_state, 0.0, bool(done)
        
        # Fallback to Python implementation
        # Fast copy using list() instead of .copy()
        new_board = list(state.board)
        new_deck = list(state.deck)
        
        if state.round == 0:
            # Place all 5 cards
            for card_idx, slot_idx in action.placements:
                new_board[slot_idx] = state.current_draw[card_idx]
            
            # Deal next 3 cards for round 1
            if len(new_deck) >= 3:
                next_draw = [new_deck.pop(), new_deck.pop(), new_deck.pop()]
                new_round = 1
            else:
                next_draw = []
                new_round = 5  # End game (round > 4)
        else:
            # Place 2 of 3 cards
            kept_cards = [state.current_draw[i] for i in action.keep_indices]
            for keep_idx, slot_idx in action.placements:
                new_board[slot_idx] = kept_cards[keep_idx]
            
            # Move to next round
            new_round = state.round + 1
            
            # Deal next 3 cards if not done
            if new_round <= 4 and len(new_deck) >= 3:
                next_draw = [new_deck.pop(), new_deck.pop(), new_deck.pop()]
            else:
                next_draw = []
        
        # Fast completion check
        done = new_round > 4 or all(slot is not None for slot in new_board)
        
        new_state = State(
            board=new_board,
            round=new_round,
            current_draw=next_draw,
            deck=new_deck,
            cards_placed_this_round=len(action.placements)
        )
        
        return new_state, 0.0, done
    
    def score(self, state: State) -> float:
        """
        Compute final OFC score for a completed board.
        Uses proper OFC scoring with royalties and foul penalties.
        Returns total score (royalties - foul penalty if applicable).
        """
        if not all(slot is not None for slot in state.board):
            # Board not complete, return 0 or negative
            return 0.0
        
        bottom_cards = [state.board[i] for i in range(5)]
        middle_cards = [state.board[i] for i in range(5, 10)]
        top_cards = [state.board[i] for i in range(10, 13)]
        
        # Try C++ accelerated scoring if available
        if _CPP is not None:
            import numpy as _np
            b = _np.array([c.to_int() for c in bottom_cards], dtype=_np.int16)
            m = _np.array([c.to_int() for c in middle_cards], dtype=_np.int16)
            t = _np.array([c.to_int() for c in top_cards], dtype=_np.int16)
            s, fouled = _CPP.score_board_from_ints(b, m, t)
            return float(s)
        
        # Fallback to Python scoring
        score, is_fouled = score_board(bottom_cards, middle_cards, top_cards)
        return score

