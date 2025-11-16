#pragma once

#include "Deck.hpp"
#include <array>
#include <optional>
#include <string>
#include <vector>

namespace ofc {

struct PlayerBoard {
    std::vector<Card> top;    // 3 cards
    std::vector<Card> middle; // 5 cards
    std::vector<Card> bottom; // 5 cards
    std::vector<Card> hand;   // current cards to place
    std::vector<Card> discards;
    bool inFantasyland = false;
    mutable bool stayInFantasy = false;

    void reset();
};

struct MatchResult {
    int points = 0;
    bool foul = false;
};

class GameState {
public:
    GameState();

    void newHand(unsigned int seed);
    bool dealInitial(bool fantasy);
    bool dealNext(bool fantasy);
    
    // Remove specific cards from the deck (used for Monte Carlo simulation)
    void removeCardsFromDeck(const std::vector<Card>& cards);

    PlayerBoard& bot();
    const PlayerBoard& bot() const;

    void applyPlacements(const std::vector<std::pair<std::string, Card>>& placements,
                         const std::optional<Card>& discard);

    bool isComplete() const;

    MatchResult scoreBot() const;

    void finalizeHand();
    bool botInFantasyland() const;
    bool botStayInFantasy() const;

private:
    Deck deck_;
    PlayerBoard bot_;
    int deals_;
    bool fantasyNext_;
};

} // namespace ofc

