#pragma once

#include "Deck.hpp"
#include "HandEvaluator.hpp"
#include "Rules.hpp"
#include <string>
#include <vector>

namespace ofc {

struct BoardView {
    const std::vector<Card>& top;
    const std::vector<Card>& middle;
    const std::vector<Card>& bottom;
};

struct ValidationResult {
    bool fouled = false;
    std::string reason;
};

ValidationResult validateBoard(const BoardView& board);
int royaltiesTop(const std::vector<Card>& top);
int royaltiesFive(const std::vector<Card>& cards, bool isMiddle);
int totalRoyalties(const BoardView& board);
int settlePairwise(const BoardView& a, const BoardView& b);
bool checkFantasylandEligibility(const BoardView& board, bool validateFouls = true);
bool checkFantasylandContinuation(const BoardView& board);

} // namespace ofc

