#pragma once

#include <array>
#include <string>
#include <vector>

namespace ofc {

using Card = std::string; // e.g. "AH", "Td"
using DeckArray = std::array<Card, 52>;

class Deck {
public:
    Deck();

    void shuffle(unsigned int seed);
    Card deal();
    std::size_t remaining() const;
    
    // Remove specific cards from the deck (used for simulation)
    void removeCards(const std::vector<Card>& cards);

private:
    DeckArray cards_;
    std::size_t index_;
};

std::vector<Card> dealCards(Deck& deck, std::size_t count);
std::vector<Card> generateStandardDeck();

} // namespace ofc

