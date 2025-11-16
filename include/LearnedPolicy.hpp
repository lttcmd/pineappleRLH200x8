#pragma once

#include "BotPolicy.hpp"
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>

namespace ofc {

class LearnedPolicy : public BotPolicy {
public:
    LearnedPolicy(std::string tablePath,
                  double foulThreshold = 0.12,
                  std::unique_ptr<BotPolicy> fallback = nullptr);

    Move chooseInitial(const GameState& state) override;
    Move chooseTurn(const GameState& state) override;

    const std::string& tablePath() const { return tablePath_; }
    bool loaded() const { return !table_.empty(); }

private:
    struct Entry {
        Move move;
        double avgScore = 0.0;
        double foulRate = 0.0;
        int samples = 0;
    };

    std::optional<Move> lookup(const GameState& state) const;
    bool loadTable(const std::string& path);

    std::string tablePath_;
    double foulThreshold_;
    std::unique_ptr<BotPolicy> fallback_;
    std::unordered_map<std::string, Entry> table_;
};

} // namespace ofc


