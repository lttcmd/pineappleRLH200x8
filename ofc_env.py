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
        legal = []
        
        if state.round == 0:
            # Initial round: must place all 5 cards
            # Find empty slots (optimized with list comp)
            empty_slots = [i for i in range(13) if state.board[i] is None]
            
            if len(empty_slots) < 5:
                return []  # Invalid state
            
            # Drastically reduced permutations - only generate first 24 (was 120, then 60)
            count = 0
            for perm in permutations(range(5)):
                if count >= 24:  # Minimal permutations for maximum speed
                    break
                placements = [(card_idx, empty_slots[i]) for i, card_idx in enumerate(perm)]
                legal.append(Action(keep_indices=tuple(range(5)), placements=placements))
                count += 1
        else:
            # Rounds 1-4: choose 2 of 3 cards
            empty_slots = [i for i in range(13) if state.board[i] is None]
            
            if len(empty_slots) < 2:
                return []  # Board is full
            
            # Choose 2 of 3 cards (3 combinations: (0,1), (0,2), (1,2))
            for keep in combinations(range(3), 2):
                # For each pair of cards, try placing them in empty slots
                # Further limit slot combinations for maximum speed
                slot_combos = list(combinations(empty_slots, 2))
                for slot_pair in slot_combos[:min(len(slot_combos), 15)]:  # Reduced from 20 to 15
                    # Both orderings
                    legal.append(Action(keep_indices=keep, placements=[(0, slot_pair[0]), (1, slot_pair[1])]))
                    legal.append(Action(keep_indices=keep, placements=[(1, slot_pair[0]), (0, slot_pair[1])]))
        
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
        
        # Use proper OFC scoring
        score, is_fouled = score_board(bottom_cards, middle_cards, top_cards)
        return score

