#pragma once

#include "BotPolicy.hpp"
#include "GameState.hpp"
#include <string>
#include <vector>

namespace ofc {

struct NamedFeature {
    std::string name;
    double value = 0.0;
};

struct FeatureVector {
    std::string bucketId;
    std::vector<NamedFeature> metrics;
};

FeatureVector extractFeatures(const GameState& state, const Move& move);
std::string summarizeMove(const Move& move);
std::string bucketForState(const GameState& state);
Move parseMoveSignature(const std::string& signature);

} // namespace ofc

