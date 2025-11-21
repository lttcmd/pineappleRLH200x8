#pragma once

namespace ofc::rules {

// Board sizes
constexpr int TOP_SIZE = 3;
constexpr int MIDDLE_SIZE = 5;
constexpr int BOTTOM_SIZE = 5;

// Foul penalty (negative score for fouling)
constexpr int FOUL_PENALTY = 6;

// Royalty tables for bottom and middle rows
struct RoyaltyTable {
  int straight;
  int flush;
  int fullHouse;
  int quads;
  int straightFlush;
  int royalFlush;
  int trips;  // Only for bottom row
};

inline RoyaltyTable bottomRoyalties() {
  return {
    .straight = 2,
    .flush = 4,
    .fullHouse = 6,
    .quads = 10,
    .straightFlush = 15,
    .royalFlush = 25,
    .trips = 0
  };
}

inline RoyaltyTable middleRoyalties() {
  return {
    .straight = 4,
    .flush = 8,
    .fullHouse = 12,
    .quads = 20,
    .straightFlush = 30,
    .royalFlush = 50,
    .trips = 2
  };
}

}  // namespace ofc::rules

