"""
OFC Scoring implementation matching the C++ Rules.hpp and Scoring.hpp
"""
from typing import List, Tuple
from ofc_types import Card, Rank, Suit


# Constants from Rules.hpp
TOP_SIZE = 3
MIDDLE_SIZE = 5
BOTTOM_SIZE = 5
INITIAL_SET_COUNT = 5
ROUNDS = 4
CARDS_PER_ROUND = 3
PLACE_COUNT_PER_ROUND = 2
ROW_WIN = 1
SCOOP_BONUS = 3
FOUL_PENALTY = 6


def rank_to_string(rank: Rank) -> str:
    """Convert rank to string representation (2-9, T, J, Q, K, A)."""
    if rank == Rank.TEN:
        return "T"
    elif rank == Rank.JACK:
        return "J"
    elif rank == Rank.QUEEN:
        return "Q"
    elif rank == Rank.KING:
        return "K"
    elif rank == Rank.ACE:
        return "A"
    else:
        return str(rank.value)


def top_has_pair(cards: List[Card]) -> bool:
    """Check if top row has a pair (regardless of royalty)."""
    if len(cards) != 3:
        return False
    ranks = [c.rank for c in cards]
    rank_counts = {}
    for rank in ranks:
        rank_counts[rank] = rank_counts.get(rank, 0) + 1
    return any(count == 2 for count in rank_counts.values())


def top_pair_royalty(cards: List[Card]) -> int:
    """Calculate top row pair royalty (Rules.hpp topPairRoyalty)."""
    if len(cards) != 3:
        return 0
    
    ranks = [c.rank for c in cards]
    rank_counts = {}
    for rank in ranks:
        rank_counts[rank] = rank_counts.get(rank, 0) + 1
    
    # Check for pair
    for rank, count in rank_counts.items():
        if count == 2:
            rank_str = rank_to_string(rank)
            royalty_table = {
                "66": 1, "77": 2, "88": 3, "99": 4,
                "TT": 5, "JJ": 6, "QQ": 7, "KK": 8, "AA": 9
            }
            key = rank_str + rank_str
            return royalty_table.get(key, 0)
    
    return 0


def top_trips_royalty(cards: List[Card]) -> int:
    """Calculate top row trips royalty (Rules.hpp topTripsRoyalty)."""
    if len(cards) != 3:
        return 0
    
    ranks = [c.rank for c in cards]
    if len(set(ranks)) == 1:  # All same rank (trips)
        rank_str = rank_to_string(ranks[0])
        royalty_table = {
            "222": 10, "333": 11, "444": 12, "555": 13, "666": 14,
            "777": 15, "888": 16, "999": 17, "TTT": 18, "JJJ": 19,
            "QQQ": 20, "KKK": 21, "AAA": 22
        }
        key = rank_str * 3
        return royalty_table.get(key, 0)
    
    return 0


def royalties_top(cards: List[Card]) -> int:
    """Calculate total royalties for top row (3 cards)."""
    return max(top_pair_royalty(cards), top_trips_royalty(cards))


def royalties_five(cards: List[Card], is_middle: bool) -> int:
    """
    Calculate royalties for 5-card hand (middle or bottom row).
    Matches Rules.hpp middleRoyalties() and bottomRoyalties().
    """
    if len(cards) != 5:
        return 0
    
    royalties = 0
    
    # Determine which royalty table to use
    if is_middle:
        # Middle royalties
        straight_royalty = 4
        flush_royalty = 8
        full_house_royalty = 12
        quads_royalty = 20
        straight_flush_royalty = 30
        royal_flush_royalty = 50
        trips_royalty = 2
    else:
        # Bottom royalties
        straight_royalty = 2
        flush_royalty = 4
        full_house_royalty = 6
        quads_royalty = 10
        straight_flush_royalty = 15
        royal_flush_royalty = 25
        trips_royalty = 0
    
    # Evaluate hand type
    hand_type = evaluate_5card_hand(cards)
    
    if hand_type == "royal_flush":
        royalties = royal_flush_royalty
    elif hand_type == "straight_flush":
        royalties = straight_flush_royalty
    elif hand_type == "quads":
        royalties = quads_royalty
    elif hand_type == "full_house":
        royalties = full_house_royalty
    elif hand_type == "flush":
        royalties = flush_royalty
    elif hand_type == "straight":
        royalties = straight_royalty
    elif hand_type == "trips" and is_middle:
        royalties = trips_royalty
    
    return royalties


def evaluate_5card_hand(cards: List[Card]) -> str:
    """
    Evaluate a 5-card hand and return hand type.
    Returns: "royal_flush", "straight_flush", "quads", "full_house", 
             "flush", "straight", "trips", "two_pair", "pair", "high_card"
    """
    ranks = sorted([c.rank.value for c in cards], reverse=True)
    suits = [c.suit for c in cards]
    
    # Count rank frequencies
    rank_counts = {}
    for rank in ranks:
        rank_counts[rank] = rank_counts.get(rank, 0) + 1
    
    counts = sorted(rank_counts.values(), reverse=True)
    is_flush = len(set(suits)) == 1
    is_straight = is_straight_5(ranks)
    is_royal = is_royal_flush(cards)
    
    # Royal flush
    if is_royal:
        return "royal_flush"
    
    # Straight flush
    if is_straight and is_flush:
        return "straight_flush"
    
    # Four of a kind
    if counts == [4, 1]:
        return "quads"
    
    # Full house
    if counts == [3, 2]:
        return "full_house"
    
    # Flush
    if is_flush:
        return "flush"
    
    # Straight
    if is_straight:
        return "straight"
    
    # Three of a kind
    if counts == [3, 1, 1]:
        return "trips"
    
    # Two pair
    if counts == [2, 2, 1]:
        return "two_pair"
    
    # One pair
    if counts == [2, 1, 1, 1]:
        return "pair"
    
    # High card
    return "high_card"


def is_royal_flush(cards: List[Card]) -> bool:
    """Check if cards form a royal flush (A, K, Q, J, T all same suit)."""
    if len(cards) != 5:
        return False
    
    suits = [c.suit for c in cards]
    ranks = {c.rank for c in cards}
    
    required_ranks = {Rank.ACE, Rank.KING, Rank.QUEEN, Rank.JACK, Rank.TEN}
    
    return len(set(suits)) == 1 and ranks == required_ranks


def is_straight_5(ranks: List[int]) -> bool:
    """Check if 5 ranks form a straight."""
    unique_ranks = sorted(set(ranks))
    
    if len(unique_ranks) != 5:
        return False
    
    # Check for A-2-3-4-5 (wheel)
    if unique_ranks == [2, 3, 4, 5, 14]:
        return True
    
    # Check regular straight
    for i in range(len(unique_ranks) - 1):
        if unique_ranks[i + 1] - unique_ranks[i] != 1:
            return False
    
    return True


def compare_5card_hands(hand1: List[Card], hand2: List[Card]) -> int:
    """
    Compare two 5-card hands.
    Returns: 1 if hand1 > hand2, -1 if hand1 < hand2, 0 if equal
    """
    type1 = evaluate_5card_hand(hand1)
    type2 = evaluate_5card_hand(hand2)
    
    # Hand type hierarchy
    type_order = {
        "royal_flush": 9,
        "straight_flush": 8,
        "quads": 7,
        "full_house": 6,
        "flush": 5,
        "straight": 4,
        "trips": 3,
        "two_pair": 2,
        "pair": 1,
        "high_card": 0
    }
    
    order1 = type_order.get(type1, 0)
    order2 = type_order.get(type2, 0)
    
    if order1 > order2:
        return 1
    elif order1 < order2:
        return -1
    
    # Same type, compare by rank values
    ranks1 = sorted([c.rank.value for c in hand1], reverse=True)
    ranks2 = sorted([c.rank.value for c in hand2], reverse=True)
    
    # For same hand type, compare kickers
    return compare_ranks(ranks1, ranks2)


def compare_ranks(ranks1: List[int], ranks2: List[int]) -> int:
    """Compare two rank lists lexicographically."""
    for r1, r2 in zip(ranks1, ranks2):
        if r1 > r2:
            return 1
        elif r1 < r2:
            return -1
    return 0


def compare_3card_hands(hand1: List[Card], hand2: List[Card]) -> int:
    """
    Compare two 3-card hands (top row).
    Returns: 1 if hand1 > hand2, -1 if hand1 < hand2, 0 if equal
    """
    if len(hand1) != 3 or len(hand2) != 3:
        return 0
    
    ranks1 = sorted([c.rank.value for c in hand1], reverse=True)
    ranks2 = sorted([c.rank.value for c in hand2], reverse=True)
    
    # Count pairs/trips
    counts1 = {}
    counts2 = {}
    for r in ranks1:
        counts1[r] = counts1.get(r, 0) + 1
    for r in ranks2:
        counts2[r] = counts2.get(r, 0) + 1
    
    # Trips > pair > high card
    max_count1 = max(counts1.values())
    max_count2 = max(counts2.values())
    
    if max_count1 > max_count2:
        return 1
    elif max_count1 < max_count2:
        return -1
    
    # Same hand type, compare ranks
    return compare_ranks(ranks1, ranks2)


def validate_board(bottom: List[Card], middle: List[Card], top: List[Card]) -> Tuple[bool, str]:
    """
    Validate that board follows OFC rules (bottom > middle > top).
    Returns: (is_valid, reason)
    """
    if len(bottom) != 5 or len(middle) != 5 or len(top) != 3:
        return False, "Incorrect number of cards"
    
    # Check bottom > middle (both are 5-card hands)
    bottom_vs_middle = compare_5card_hands(bottom, middle)
    if bottom_vs_middle <= 0:
        return False, "Bottom hand must beat middle hand"
    
    # Check middle > top
    # In OFC, we compare the 5-card middle hand against the 3-card top hand
    # The rule: middle must be stronger. We do this by comparing hand categories.
    # A 5-card hand generally beats a 3-card hand of the same category,
    # but we need to check if top's hand type would beat middle's.
    
    middle_type = evaluate_5card_hand(middle)
    top_has_pair_flag = top_has_pair(top)
    top_pair_roy = top_pair_royalty(top)
    top_trips_roy = top_trips_royalty(top)
    
    # Top has trips
    if top_trips_roy > 0:
        # Middle must have at least a full house to beat trips in top
        if middle_type not in ["full_house", "quads", "straight_flush", "royal_flush"]:
            # Check if middle's trips are higher
            middle_ranks = sorted([c.rank.value for c in middle], reverse=True)
            top_rank = sorted([c.rank.value for c in top], reverse=True)[0]
            # If middle is just trips, compare ranks
            if middle_type == "trips":
                if max(middle_ranks) <= top_rank:
                    return False, "Middle hand must beat top hand"
            elif middle_type in ["high_card", "pair", "two_pair"]:
                return False, "Middle hand must beat top hand"
    
    # Top has pair (check for ANY pair, not just royalty pairs)
    elif top_has_pair_flag:
        # Middle must have at least trips to beat a pair in top
        if middle_type in ["high_card", "pair", "two_pair"]:
            return False, "Middle hand must beat top hand"
        # If middle is trips, it's automatically stronger
        # If middle is full house or better, it's stronger
    
    # Top is high card
    else:
        # Middle should generally be stronger, but high card vs high card is tricky
        # For simplicity, if middle is also high card, compare highest cards
        if middle_type == "high_card":
            middle_ranks = sorted([c.rank.value for c in middle], reverse=True)
            top_ranks = sorted([c.rank.value for c in top], reverse=True)
            if middle_ranks[0] <= top_ranks[0]:
                return False, "Middle hand must beat top hand"
    
    return True, "Valid"


def total_royalties(bottom: List[Card], middle: List[Card], top: List[Card]) -> int:
    """Calculate total royalties for a complete board."""
    return (royalties_five(bottom, is_middle=False) + 
            royalties_five(middle, is_middle=True) + 
            royalties_top(top))


def score_board(bottom: List[Card], middle: List[Card], top: List[Card]) -> Tuple[float, bool]:
    """
    Score a complete board.
    
    Scoring rules:
    - If fouled (bottom <= middle or middle <= top): -6 points
    - If not fouled: royalties only (0 if no special hands, positive if royalties exist)
    
    Note: ROW_WIN (+1 per row won) and SCOOP_BONUS (+3) are for multiplayer pairwise
    comparison only. In single-player training, we use royalties only.
    
    Returns: (score, is_fouled)
    """
    is_valid, reason = validate_board(bottom, middle, top)
    
    if not is_valid:
        # Fouled: -6 point penalty
        return -FOUL_PENALTY, True
    
    # Not fouled: Calculate royalties
    # Royalties can be 0 (no special hands) or positive (pairs, trips, straights, etc.)
    royalties = total_royalties(bottom, middle, top)
    
    # In single-player mode: score = royalties (no base points)
    # In multiplayer: would add +1 per row won +3 for scoop
    return float(royalties), False

