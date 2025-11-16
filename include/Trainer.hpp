#pragma once

#include "BotPolicy.hpp"
#include "GameState.hpp"
#include "TrainingLogger.hpp"
#include <cstdint>
#include <memory>

namespace ofc {

struct TrainingStats {
    std::uint64_t handsPlayed = 0;
    std::uint64_t wins = 0;
    std::uint64_t fouls = 0;
    double totalScore = 0.0;
};

class Trainer {
public:
    Trainer(std::unique_ptr<BotPolicy> policy, TrainingLogger* logger = nullptr);

    void setLogger(TrainingLogger* logger);
    TrainingStats run(std::uint64_t hands, unsigned int seed);

private:
    std::unique_ptr<BotPolicy> policy_;
    TrainingLogger* logger_;
    GameState state_;
};

} // namespace ofc

