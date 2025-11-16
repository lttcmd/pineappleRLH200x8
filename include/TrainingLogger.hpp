#pragma once

#include "FeatureExtractor.hpp"
#include <cstdint>
#include <fstream>
#include <mutex>
#include <string>

namespace ofc {

struct TrainingSample {
    std::uint64_t handId = 0;
    std::uint32_t decisionIndex = 0;
    std::string stage;
    FeatureVector features;
    std::string moveSignature;
    double score = 0.0;
    bool foul = false;
};

class TrainingLogger {
public:
    explicit TrainingLogger(const std::string& path);
    void log(const TrainingSample& sample);

private:
    void writeHeader(const TrainingSample& sample);

    std::ofstream stream_;
    std::mutex mutex_;
    bool headerWritten_ = false;
};

} // namespace ofc

