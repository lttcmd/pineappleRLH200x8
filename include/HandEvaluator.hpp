#pragma once

#include "Deck.hpp"
#include <array>
#include <vector>

namespace ofc {

struct Hand5 {
    int category;
    std::array<int, 5> key;
};

struct Hand3 {
    int category;
    std::array<int, 3> key;
};

Hand5 rank5(const std::vector<Card>& cards);
Hand3 rankTop3(const std::vector<Card>& cards);
int compare5(const Hand5& a, const Hand5& b);
int compareTop3(const Hand3& a, const Hand3& b);
bool isRoyalFlush(const std::vector<Card>& cards);

} // namespace ofc

