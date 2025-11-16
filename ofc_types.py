"""
Shared types for OFC implementation.
"""
from dataclasses import dataclass
from enum import Enum


class Suit(Enum):
    SPADES = 0
    HEARTS = 1
    DIAMONDS = 2
    CLUBS = 3


class Rank(Enum):
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14


@dataclass
class Card:
    """Represents a playing card."""
    rank: Rank
    suit: Suit
    
    def __hash__(self):
        return hash((self.rank, self.suit))
    
    def __eq__(self, other):
        return isinstance(other, Card) and self.rank == other.rank and self.suit == other.suit
    
    def to_int(self) -> int:
        """Convert card to integer 0-51 for encoding."""
        return (self.rank.value - 2) * 4 + self.suit.value

