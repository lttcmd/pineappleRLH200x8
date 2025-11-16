#pragma once

#include "BotPolicy.hpp"
#include <optional>
#include <random>
#include <vector>

namespace ofc {

class MonteCarloPolicy : public BotPolicy {
public:
    MonteCarloPolicy(int rolloutsInitial = 40, int rolloutsTurn = 30, double foulThreshold = 0.12);

    Move chooseInitial(const GameState& state) override;
    Move chooseTurn(const GameState& state) override;

private:
    struct EvaluatedMove {
        Move move;
        double avgScore = 0.0;
        double foulRate = 0.0;
    };

    int rolloutsInitial_;
    int rolloutsTurn_;
    double foulThreshold_;

    std::mt19937 rng_;

    std::vector<Move> generateInitialCandidates(const PlayerBoard& board) const;
    std::vector<Move> generateTurnCandidates(const GameState& state) const;
    EvaluatedMove evaluateCandidate(const GameState& state, const Move& move, int rollouts);
    EvaluatedMove selectBest(const GameState& state, const std::vector<Move>& moves, int rollouts, const char* context);
};

} // namespace ofc

