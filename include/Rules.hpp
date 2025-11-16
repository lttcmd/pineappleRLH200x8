#pragma once

#include <string>
#include <unordered_map>

namespace ofc::rules {

constexpr int TOP_SIZE = 3;
constexpr int MIDDLE_SIZE = 5;
constexpr int BOTTOM_SIZE = 5;

constexpr int INITIAL_SET_COUNT = 5;
constexpr int ROUNDS = 4;
constexpr int CARDS_PER_ROUND = 3;
constexpr int PLACE_COUNT_PER_ROUND = 2;

constexpr int ROW_WIN = 1;
constexpr int SCOOP_BONUS = 3;
constexpr int FOUL_PENALTY = 6;
constexpr bool PUSHES_ALLOWED = true;

inline int topPairRoyalty(const std::string& key) {
    static const std::unordered_map<std::string, int> table{
        {"66", 1}, {"77", 2}, {"88", 3}, {"99", 4},
        {"TT", 5}, {"JJ", 6}, {"QQ", 7}, {"KK", 8}, {"AA", 9}
    };
    if (auto it = table.find(key); it != table.end()) {
        return it->second;
    }
    return 0;
}

inline int topTripsRoyalty(const std::string& key) {
    static const std::unordered_map<std::string, int> table{
        {"222", 10}, {"333", 11}, {"444", 12}, {"555", 13}, {"666", 14},
        {"777", 15}, {"888", 16}, {"999", 17}, {"TTT", 18}, {"JJJ", 19},
        {"QQQ", 20}, {"KKK", 21}, {"AAA", 22}
    };
    if (auto it = table.find(key); it != table.end()) {
        return it->second;
    }
    return 0;
}

struct FiveCardRoyalties {
    int straight = 0;
    int flush = 0;
    int fullHouse = 0;
    int quads = 0;
    int straightFlush = 0;
    int royalFlush = 0;
    int trips = 0; // used for middle row
};

inline const FiveCardRoyalties& middleRoyalties() {
    static const FiveCardRoyalties tbl{
        .straight = 4,
        .flush = 8,
        .fullHouse = 12,
        .quads = 20,
        .straightFlush = 30,
        .royalFlush = 50,
        .trips = 2
    };
    return tbl;
}

inline const FiveCardRoyalties& bottomRoyalties() {
    static const FiveCardRoyalties tbl{
        .straight = 2,
        .flush = 4,
        .fullHouse = 6,
        .quads = 10,
        .straightFlush = 15,
        .royalFlush = 25,
        .trips = 0
    };
    return tbl;
}

namespace fantasyland {
    constexpr bool ENABLED = true;
    constexpr const char* TOP_PAIR_AT_LEAST = "QQ";
    constexpr bool TOP_TRIPS_OR_BETTER = true;
    constexpr bool NO_FOUL_REQUIRED = true;
    constexpr int HAND_SIZE = 14;
}

} // namespace ofc::rules

