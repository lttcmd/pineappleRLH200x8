#pragma once

#include "GameState.hpp"
#include <optional>
#include <vector>

namespace ofc {

struct Move {
    std::vector<std::pair<std::string, Card>> placements;
    std::optional<Card> discard;
};

class BotPolicy {
public:
    virtual ~BotPolicy() = default;

    virtual Move chooseInitial(const GameState& state) = 0;
    virtual Move chooseTurn(const GameState& state) = 0;
};

class RandomPolicy : public BotPolicy {
public:
    Move chooseInitial(const GameState& state) override;
    Move chooseTurn(const GameState& state) override;
};

class HeuristicRolloutPolicy : public BotPolicy {
public:
    Move chooseInitial(const GameState& state) override;
    Move chooseTurn(const GameState& state) override;
};

} // namespace ofc

