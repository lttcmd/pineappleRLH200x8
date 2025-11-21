#include "sfl_policy.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include "Rules.hpp"

#ifdef _WIN32
typedef Py_ssize_t ssize_t;
#endif

namespace py = pybind11;

py::array_t<int16_t> legal_actions_round0(py::array_t<int16_t> board);
py::tuple legal_actions_rounds1to4(py::array_t<int16_t> board, int round_idx);
py::tuple step_state_round0(py::array_t<int16_t> board,
                            py::array_t<int16_t> current5,
                            py::array_t<int16_t> deck,
                            py::array_t<int16_t> slots5);
py::tuple step_state(py::array_t<int16_t> board,
                     int round_idx,
                     py::array_t<int16_t> current_draw,
                     py::array_t<int16_t> deck,
                     int keep_i, int keep_j,
                     int place0_keep_idx, int place0_slot,
                     int place1_keep_idx, int place1_slot);
py::array_t<float> encode_state_batch_ints(py::array_t<int16_t> boards,
                                           py::array_t<int16_t> rounds,
                                           py::array_t<int16_t> draws,
                                           py::array_t<int16_t> deck_sizes);

// Canonical OFC single-board scorer (royalties / foul penalty) from ofc_score.cpp.
std::pair<float, bool> score_board_from_ints(py::array_t<int16_t> bottom,
                                             py::array_t<int16_t> middle,
                                             py::array_t<int16_t> top);

namespace {

// Number of rollout samples used per round when evaluating an action.
// Round 0 is the most important street: we double its samples to get a
// less noisy estimate of long-term potential from the initial layout.
constexpr std::array<int, 5> kSamplesPerRound = {96, 36, 24, 16, 8};
constexpr std::array<int, 5> kSamplesPerRoundFast = {2, 2, 1, 1, 1};

thread_local bool g_force_fast_samples = false;

struct FastSamplesGuard {
  bool prev;
  FastSamplesGuard(bool enable) : prev(g_force_fast_samples) {
    if (enable) g_force_fast_samples = true;
  }
  ~FastSamplesGuard() { g_force_fast_samples = prev; }
};

struct SflShapingConfig {
  float foul_penalty;   // planning-time reward for a foul (negative)
  float pass_penalty;   // planning-time reward for a zero-royalty pass (negative)
  float medium_bonus;   // added to royalties for 4 <= r < 8
  float strong_bonus;   // added to royalties for 8 <= r < 12
  float monster_mult;   // multiplier applied to royalties when r >= 12
};

// Default shaping chosen from parameter sweep ("sf_base") as a stable,
// royalty-seeking baseline that RL can improve on. We slightly upweight
// monster hands (monster_mult=10) so obviously huge boards are preferred
// in planning when they are available.
static SflShapingConfig g_sfl_cfg{
    /*foul_penalty=*/-4.0f,
    /*pass_penalty=*/-3.0f,
    /*medium_bonus=*/4.0f,
    /*strong_bonus=*/8.0f,
    /*monster_mult=*/10.0f};

struct PartialBoard {
  std::vector<int16_t> bottom;
  std::vector<int16_t> middle;
  std::vector<int16_t> top;
};

enum class HandType {
  kHighCard = 0,
  kPair = 1,
  kTwoPair = 2,
  kTrips = 3,
  kStraight = 4,
  kFlush = 5,
  kFullHouse = 6,
  kQuads = 7,
  kStraightFlush = 8,
  kRoyalFlush = 9
};

inline int rank_from_card(int16_t c) { return (c / 4) + 2; }
inline int suit_from_card(int16_t c) { return c % 4; }

std::array<int16_t, 13> board_from_numpy(py::array_t<int16_t>& board_np) {
  auto view = board_np.unchecked<1>();
  std::array<int16_t, 13> board{};
  for (int i = 0; i < 13; ++i) board[i] = view(i);
  return board;
}

std::vector<int16_t> deck_from_numpy(py::array_t<int16_t>& deck_np) {
  auto view = deck_np.unchecked<1>();
  std::vector<int16_t> deck(view.shape(0));
  for (ssize_t i = 0; i < view.shape(0); ++i) deck[i] = view(i);
  return deck;
}

PartialBoard extract_rows(const std::array<int16_t, 13>& board) {
  PartialBoard pb;
  for (int i = 0; i < 5; ++i) if (board[i] >= 0) pb.bottom.push_back(board[i]);
  for (int i = 5; i < 10; ++i) if (board[i] >= 0) pb.middle.push_back(board[i]);
  for (int i = 10; i < 13; ++i) if (board[i] >= 0) pb.top.push_back(board[i]);
  return pb;
}

template <size_t N>
std::array<int16_t, N> merge_row(const std::vector<int16_t>& base,
                                 const std::vector<int16_t>& add) {
  std::array<int16_t, N> result{};
  std::vector<int16_t> merged = base;
  merged.insert(merged.end(), add.begin(), add.end());
  if (merged.size() != N) {
    throw std::runtime_error("merge_row received mismatched sizes");
  }
  for (size_t i = 0; i < N; ++i) result[i] = merged[i];
  return result;
}

bool is_wheel(const std::array<int, 5>& ranks_sorted) {
  return ranks_sorted[0] == 2 && ranks_sorted[1] == 3 && ranks_sorted[2] == 4 &&
         ranks_sorted[3] == 5 && ranks_sorted[4] == 14;
}

HandType classify_hand(const std::array<int16_t, 5>& cards) {
  std::array<int, 5> ranks{};
  std::array<int, 5> suits{};
  for (int i = 0; i < 5; ++i) {
    ranks[i] = rank_from_card(cards[i]);
    suits[i] = suit_from_card(cards[i]);
  }
  std::array<int, 5> ranks_sorted = ranks;
  std::sort(ranks_sorted.begin(), ranks_sorted.end());
  bool flush = (suits[0] == suits[1] && suits[1] == suits[2] &&
                suits[2] == suits[3] && suits[3] == suits[4]);
  bool straight = false;
  if (is_wheel(ranks_sorted)) {
    straight = true;
  } else {
    straight = true;
    for (int i = 0; i < 4; ++i) {
      if (ranks_sorted[i + 1] - ranks_sorted[i] != 1) {
        straight = false;
        break;
      }
    }
  }

  std::array<int, 15> counts{};
  counts.fill(0);
  for (int r : ranks) counts[r]++;
  std::vector<int> freq;
  freq.reserve(5);
  for (int c : counts) if (c > 0) freq.push_back(c);
  std::sort(freq.begin(), freq.end(), std::greater<int>());

  if (straight && flush) {
    if (ranks_sorted[0] == 10 && ranks_sorted[4] == 14) {
      return HandType::kRoyalFlush;
    }
    return HandType::kStraightFlush;
  }
  if (!freq.empty() && freq[0] == 4) return HandType::kQuads;
  if (freq.size() >= 2 && freq[0] == 3 && freq[1] == 2) return HandType::kFullHouse;
  if (flush) return HandType::kFlush;
  if (straight) return HandType::kStraight;
  if (!freq.empty() && freq[0] == 3) return HandType::kTrips;
  if (freq.size() >= 2 && freq[0] == 2 && freq[1] == 2) return HandType::kTwoPair;
  if (!freq.empty() && freq[0] == 2) return HandType::kPair;
  return HandType::kHighCard;
}

int hand_rank_value(HandType t) {
  return static_cast<int>(t);
}

int five_card_royalty(HandType type, bool is_middle) {
  const auto& tbl = is_middle ? ofc::rules::middleRoyalties()
                              : ofc::rules::bottomRoyalties();
  switch (type) {
    case HandType::kStraight: return tbl.straight;
    case HandType::kFlush: return tbl.flush;
    case HandType::kFullHouse: return tbl.fullHouse;
    case HandType::kQuads: return tbl.quads;
    case HandType::kStraightFlush: return tbl.straightFlush;
    case HandType::kRoyalFlush: return tbl.royalFlush;
    case HandType::kTrips: return tbl.trips;
    default: return 0;
  }
}

int top_pair_royalty(const std::array<int16_t, 3>& cards) {
  std::array<int, 3> ranks{};
  for (int i = 0; i < 3; ++i) ranks[i] = rank_from_card(cards[i]);
  std::array<int, 15> counts{};
  counts.fill(0);
  for (int r : ranks) counts[r]++;
  for (int r = 14; r >= 2; --r) {
    if (counts[r] == 2) {
      switch (r) {
        case 6: return 1;
        case 7: return 2;
        case 8: return 3;
        case 9: return 4;
        case 10: return 5;
        case 11: return 6;
        case 12: return 7;
        case 13: return 8;
        case 14: return 9;
        default: return 0;
      }
    }
  }
  return 0;
}

int top_trips_royalty(const std::array<int16_t, 3>& cards) {
  int r0 = rank_from_card(cards[0]);
  if (r0 == rank_from_card(cards[1]) && r0 == rank_from_card(cards[2])) {
    switch (r0) {
      case 2: return 10;
      case 3: return 11;
      case 4: return 12;
      case 5: return 13;
      case 6: return 14;
      case 7: return 15;
      case 8: return 16;
      case 9: return 17;
      case 10: return 18;
      case 11: return 19;
      case 12: return 20;
      case 13: return 21;
      case 14: return 22;
      default: return 0;
    }
  }
  return 0;
}

// Canonical OFC scorer: returns (royalties or -FOUL_PENALTY, fouled?) using
// the same logic as Python. This is the *true* game score.
std::pair<float, bool> canonical_score_completed_board(
    const std::array<int16_t, 5>& bottom,
    const std::array<int16_t, 5>& middle,
    const std::array<int16_t, 3>& top) {
  py::array_t<int16_t> bottom_np(py::array::ShapeContainer{5});
  py::array_t<int16_t> middle_np(py::array::ShapeContainer{5});
  py::array_t<int16_t> top_np(py::array::ShapeContainer{3});

  auto b_ptr = bottom_np.mutable_data();
  auto m_ptr = middle_np.mutable_data();
  auto t_ptr = top_np.mutable_data();
  for (int i = 0; i < 5; ++i) {
    b_ptr[i] = bottom[i];
    m_ptr[i] = middle[i];
  }
  for (int i = 0; i < 3; ++i) {
    t_ptr[i] = top[i];
  }

  return score_board_from_ints(bottom_np, middle_np, top_np);
}

// SFL-internal shaped score: slightly softer foul penalty so the heuristic
// will sometimes take high-royalty but risky lines.
//
// - Canonical scoring uses -FOUL_PENALTY (e.g. -6) for fouls.
// - Here we treat fouls as a milder -3 while keeping royalties unchanged.
//   This encourages "go for it" plays when the upside is large.
float sfl_reward_completed_board(const std::array<int16_t, 5>& bottom,
                                 const std::array<int16_t, 5>& middle,
                                 const std::array<int16_t, 3>& top) {
  auto res = canonical_score_completed_board(bottom, middle, top);
  float score = res.first;   // canonical royalties (>=0) or -FOUL_PENALTY
  bool fouled = res.second;

  // Internal foul penalty: configurable but typically slightly softer than
  // real -6. We still want fouls to hurt more than a safe pass, but not so
  // much that SFL never takes real shots.
  if (fouled) {
    return g_sfl_cfg.foul_penalty;
  }

  // For non-fouling boards, shape rewards:
  // - Actively punish completely safe, no-royalty passes (score == 0) so SFL
  //   does not "play for zero".
  // - Keep small royalties roughly neutral.
  // - Strongly upweight big royalty totals so SFL is willing to gamble when the
  //   upside is truly large. This does NOT change the real game score, only
  //   the planning objective.
  float r = std::max(0.0f, score);
  if (r <= 0.0f) {
    // Zero-royalty pass: configurable negative to discourage "just pass".
    return g_sfl_cfg.pass_penalty;
  }
  // Very small royalties (e.g. bottom straight): almost neutral.
  if (r > 0.0f && r < 4.0f) {
    return r;
  }
  // Solid but not insane (typical full houses/small flushes, etc.).
  if (r >= 4.0f && r < 8.0f) {
    return r + g_sfl_cfg.medium_bonus;
  }
  // Strong boards (bigger flushes, multiple-row royalties).
  if (r >= 8.0f && r < 12.0f) {
    return r + g_sfl_cfg.strong_bonus;
  }
  // Monster hands (quads / straight flush / huge top trips / multi-row bombs).
  if (r >= 12.0f) {
    return r + g_sfl_cfg.monster_mult * r;
  }
  // Small royalties (straights / tiny flushes) unchanged.
  return r;
}

std::vector<std::vector<int16_t>> combinations(const std::vector<int16_t>& cards,
                                               int choose) {
  std::vector<std::vector<int16_t>> combos;
  if (choose == 0) {
    combos.push_back({});
    return combos;
  }
  if (choose > static_cast<int>(cards.size())) return combos;

  std::vector<int> idx(choose);
  std::iota(idx.begin(), idx.end(), 0);
  while (true) {
    std::vector<int16_t> current;
    current.reserve(choose);
    for (int id : idx) current.push_back(cards[id]);
    combos.push_back(std::move(current));
    int i = choose - 1;
    while (i >= 0 && idx[i] == static_cast<int>(cards.size()) - choose + i) --i;
    if (i < 0) break;
    idx[i]++;
    for (int j = i + 1; j < choose; ++j) idx[j] = idx[j - 1] + 1;
  }
  return combos;
}

std::vector<int16_t> subtract_cards(const std::vector<int16_t>& cards,
                                    const std::vector<int16_t>& remove) {
  std::vector<int16_t> remaining = cards;
  for (int16_t c : remove) {
    auto it = std::find(remaining.begin(), remaining.end(), c);
    if (it != remaining.end()) remaining.erase(it);
  }
  return remaining;
}

uint64_t hash_state(const std::array<int16_t, 13>& board,
                    int round_idx,
                    const std::vector<int16_t>& deck) {
  uint64_t h = 1469598103934665603ull;
  auto mix = [&](int v) {
    h ^= static_cast<uint64_t>(static_cast<int64_t>(v) + 1);
    h *= 1099511628211ull;
  };
  mix(round_idx);
  for (int16_t c : board) mix(c);
  for (int16_t c : deck) mix(c);
  return h;
}

uint64_t mix64(uint64_t z) {
  z ^= z >> 12;
  z ^= z << 25;
  z ^= z >> 27;
  return z * 2685821657736338717ull;
}

std::vector<int16_t> sample_cards(uint64_t seed,
                                  int sample_idx,
                                  const std::vector<int16_t>& deck,
                                  int take) {
  std::vector<int16_t> pool = deck;
  std::vector<int16_t> chosen;
  chosen.reserve(take);
  uint64_t state = seed + 0x9e3779b97f4a7c15ull * static_cast<uint64_t>(sample_idx + 1);
  for (int i = 0; i < take; ++i) {
    if (pool.empty()) break;
    state = mix64(state + static_cast<uint64_t>(i + 1));
    size_t idx = static_cast<size_t>(state % pool.size());
    chosen.push_back(pool[idx]);
    pool.erase(pool.begin() + static_cast<int>(idx));
  }
  return chosen;
}

float rollout_value(const std::array<int16_t, 13>& board,
                    int next_round,
                    const std::vector<int16_t>& deck) {
  PartialBoard partial = extract_rows(board);
  int bottom_need = ofc::rules::BOTTOM_SIZE - static_cast<int>(partial.bottom.size());
  int middle_need = ofc::rules::MIDDLE_SIZE - static_cast<int>(partial.middle.size());
  int top_need = ofc::rules::TOP_SIZE - static_cast<int>(partial.top.size());
  int total_need = bottom_need + middle_need + top_need;

  if (total_need <= 0) {
    auto bottom_full = merge_row<5>(partial.bottom, {});
    auto middle_full = merge_row<5>(partial.middle, {});
    auto top_full = merge_row<3>(partial.top, {});
    return sfl_reward_completed_board(bottom_full, middle_full, top_full);
  }

  if (static_cast<int>(deck.size()) < total_need) {
    return -static_cast<float>(ofc::rules::FOUL_PENALTY);
  }

  int clamped_round = std::clamp(next_round, 0, 4);
  // Allow forcing fast sampling globally via environment as well as via
  // the thread-local guard used during dataset generation.
  static const bool kFastFromEnv = []() {
    const char* env = std::getenv("OFC_SFL_FAST");
    return env && env[0] != '\0' && env[0] != '0';
  }();
  const auto& table = (g_force_fast_samples || kFastFromEnv)
                          ? kSamplesPerRoundFast
                          : kSamplesPerRound;
  int samples = std::max(1, table[clamped_round]);
  uint64_t seed = hash_state(board, next_round, deck);

  float total_score = 0.0f;
  for (int sample_idx = 0; sample_idx < samples; ++sample_idx) {
    std::vector<int16_t> sample = sample_cards(seed, sample_idx, deck, total_need);
    if (static_cast<int>(sample.size()) < total_need) {
      total_score += -static_cast<float>(ofc::rules::FOUL_PENALTY);
      continue;
    }
    float best = -std::numeric_limits<float>::infinity();
    auto bottom_combos = combinations(sample, bottom_need);
    if (bottom_combos.empty()) bottom_combos.push_back({});
    for (const auto& bottom_add : bottom_combos) {
      auto after_bottom = subtract_cards(sample, bottom_add);
      auto middle_combos = combinations(after_bottom, middle_need);
      if (middle_combos.empty()) middle_combos.push_back({});
      for (const auto& middle_add : middle_combos) {
        auto after_middle = subtract_cards(after_bottom, middle_add);
        if (static_cast<int>(after_middle.size()) != top_need) continue;
        auto bottom_full = merge_row<5>(partial.bottom, bottom_add);
        auto middle_full = merge_row<5>(partial.middle, middle_add);
        auto top_full = merge_row<3>(partial.top, after_middle);
        float score = sfl_reward_completed_board(bottom_full, middle_full, top_full);
        if (score > best) best = score;
      }
    }
    if (best == -std::numeric_limits<float>::infinity()) {
      best = -static_cast<float>(ofc::rules::FOUL_PENALTY);
    }
    total_score += best;
  }
  return total_score / static_cast<float>(samples);
}

}  // namespace

void set_sfl_shaping(float foul_penalty,
                     float pass_penalty,
                     float medium_bonus,
                     float strong_bonus,
                     float monster_mult) {
  g_sfl_cfg.foul_penalty = foul_penalty;
  g_sfl_cfg.pass_penalty = pass_penalty;
  g_sfl_cfg.medium_bonus = medium_bonus;
  g_sfl_cfg.strong_bonus = strong_bonus;
  g_sfl_cfg.monster_mult = monster_mult;
}

int sfl_choose_action(py::array_t<int16_t> board,
                      int round_idx,
                      py::array_t<int16_t> current_draw,
                      py::array_t<int16_t> deck) {
  auto board_np = board;
  auto deck_np = deck;
  float best_score = -std::numeric_limits<float>::infinity();
  int best_idx = -1;

  if (round_idx == 0) {
    py::array_t<int16_t> placements = legal_actions_round0(board_np);
    auto placement_view = placements.unchecked<3>();
    int total = static_cast<int>(placements.shape(0));

    // Hard rule: if the initial 5 cards themselves form a strong 5-card
    // hand (straight, flush, full house, quads, straight-flush, royal),
    // then prefer an action that puts all five on the bottom row if one
    // exists. This bakes in a human-style heuristic for obvious monsters.
    if (current_draw.shape(0) == 5 && total > 0) {
      std::array<int16_t,5> initial{};
      auto cd = current_draw.unchecked<1>();
      for (int i = 0; i < 5; ++i) initial[i] = cd(i);
      HandType t = classify_hand(initial);
      bool premium =
          (t == HandType::kStraight) ||
          (t == HandType::kFlush) ||
          (t == HandType::kFullHouse) ||
          (t == HandType::kQuads) ||
          (t == HandType::kStraightFlush) ||
          (t == HandType::kRoyalFlush);
      if (premium) {
        for (int idx = 0; idx < total; ++idx) {
          bool all_bottom = true;
          for (int i = 0; i < 5; ++i) {
            int slot_idx = placement_view(idx, i, 1);
            if (slot_idx < 0 || slot_idx > 4) {
              all_bottom = false;
              break;
            }
          }
          if (all_bottom) {
            return idx;
          }
        }
      }
    }
    for (int idx = 0; idx < total; ++idx) {
      py::array_t<int16_t> slots_np(py::array::ShapeContainer{5});
      auto buf = slots_np.request();
      auto sl_ptr = static_cast<int16_t*>(buf.ptr);
      for (int i = 0; i < 5; ++i) sl_ptr[i] = -1;
      for (int i = 0; i < 5; ++i) {
        int card_idx = placement_view(idx, i, 0);
        int slot_idx = placement_view(idx, i, 1);
        if (card_idx >=0 && card_idx < 5) sl_ptr[card_idx] = slot_idx;
      }
      auto res = step_state_round0(board_np, current_draw, deck_np, slots_np);
      auto next_board = res[0].cast<py::array_t<int16_t>>();
      int next_round = res[1].cast<int>();
      auto next_deck = res[3].cast<py::array_t<int16_t>>();
      auto board_arr = board_from_numpy(next_board);
      auto deck_vec = deck_from_numpy(next_deck);
      float score = rollout_value(board_arr, next_round, deck_vec);

      // Track best overall score.
      if (score > best_score) {
        best_score = score;
        best_idx = idx;
      }
    }
    return best_idx;
  }

  py::tuple acts = legal_actions_rounds1to4(board_np, round_idx);
  auto keeps = acts[0].cast<py::array_t<int8_t>>();
  auto places = acts[1].cast<py::array_t<int16_t>>();
  auto keep_view = keeps.unchecked<2>();
  auto place_view = places.unchecked<3>();
  int total = static_cast<int>(keeps.shape(0));
  for (int idx = 0; idx < total; ++idx) {
    int keep_i = keep_view(idx, 0);
    int keep_j = keep_view(idx, 1);
    int p00 = place_view(idx, 0, 0);
    int p01 = place_view(idx, 0, 1);
    int p10 = place_view(idx, 1, 0);
    int p11 = place_view(idx, 1, 1);
    auto res = step_state(board_np, round_idx, current_draw, deck_np,
                          keep_i, keep_j, p00, p01, p10, p11);
    auto next_board = res[0].cast<py::array_t<int16_t>>();
    int next_round = res[1].cast<int>();
    auto next_deck = res[3].cast<py::array_t<int16_t>>();
    auto board_arr = board_from_numpy(next_board);
    auto deck_vec = deck_from_numpy(next_deck);
    float score = rollout_value(board_arr, next_round, deck_vec);

    // Track best overall score.
    if (score > best_score) {
      best_score = score;
      best_idx = idx;
    }
  }
  return best_idx;
}

struct SimpleState {
  std::array<int16_t,13> board;
  std::vector<int16_t> deck;
  std::array<int16_t,3> draw3;
  std::array<int16_t,5> initial5;
  int round_idx;
  bool done;
};

static void reset_state(SimpleState& st, std::mt19937_64& rng) {
  st.deck.resize(52);
  for (int i=0;i<52;++i) st.deck[i] = static_cast<int16_t>(i);
  std::shuffle(st.deck.begin(), st.deck.end(), rng);
  for (int i=0;i<13;++i) st.board[i] = -1;
  for (int i=0;i<5;++i) {
    st.initial5[i] = st.deck.back();
    st.deck.pop_back();
  }
  st.draw3 = {st.initial5[0], st.initial5[1], st.initial5[2]};
  st.round_idx = 0;
  st.done = false;
}

py::tuple generate_sfl_dataset(uint64_t seed, int num_examples) {
  const bool fast_mode = []() {
    const char* env = std::getenv("OFC_SFL_FAST");
    return env && env[0] != '\0' && env[0] != '0';
  }();
  const bool debug_mode = []() {
    const char* env = std::getenv("OFC_SFL_DEBUG");
    return env && env[0] != '\0' && env[0] != '0';
  }();

  FastSamplesGuard fast_guard(fast_mode);
  {
    py::gil_scoped_acquire gil;
    py::print("[ofc_cpp] generate_sfl_dataset start fast_mode=", fast_mode,
              " debug_mode=", debug_mode,
              " num_examples=", num_examples);
  }
  const bool verbose = debug_mode;

  std::mt19937_64 rng(seed);
  SimpleState st;
  reset_state(st, rng);

  std::vector<int16_t> boards_all;
  std::vector<int16_t> rounds_all;
  std::vector<int16_t> draws_all;
  std::vector<int16_t> deck_sizes_all;
  std::vector<int32_t> labels_all;

  std::vector<int16_t> action_boards;
  std::vector<int16_t> action_rounds;
  std::vector<int16_t> action_draws;
  std::vector<int16_t> action_deck_sizes;
  int debug_action_counter = 0;
  std::vector<int32_t> action_counts_per_state;

  boards_all.reserve(num_examples * 13);
  rounds_all.reserve(num_examples);
  draws_all.reserve(num_examples * 3);
  deck_sizes_all.reserve(num_examples);
  labels_all.reserve(num_examples);

  action_counts_per_state.reserve(num_examples);

  auto record_action_state = [&](py::array_t<int16_t>& board_arr,
                                 int next_round,
                                 py::array_t<int16_t>& draw_arr,
                                 py::array_t<int16_t>& deck_arr) {
    auto b = board_arr.unchecked<1>();
    for (int i=0;i<13;++i) action_boards.push_back(b(i));
    action_rounds.push_back(static_cast<int16_t>(next_round));
    auto d = draw_arr.unchecked<1>();
    for (int i=0;i<3;++i) {
      int16_t val = (i < d.shape(0)) ? d(i) : static_cast<int16_t>(-1);
      action_draws.push_back(val);
    }
    auto deck_view = deck_arr.unchecked<1>();
    action_deck_sizes.push_back(static_cast<int16_t>(deck_view.shape(0)));
    if (debug_mode && debug_action_counter < 4) {
      py::gil_scoped_acquire gil;
      py::list board_list;
      for (int i=0;i<13;++i) board_list.append(b(i));
      py::print("[ofc_cpp] action_state_debug board", board_list, "round", next_round);
      debug_action_counter++;
    }
  };

  for (int n=0;n<num_examples;++n) {
    auto state_start = std::chrono::steady_clock::now();
    if (st.done) {
      reset_state(st, rng);
    }
    if (verbose) {
      py::gil_scoped_acquire gil;
      py::print("[ofc_cpp] record_state", n+1, "round_idx", st.round_idx);
      if (n==0) {
        py::list init_cards;
        for (int i=0;i<5;++i) init_cards.append(st.initial5[i]);
        py::print("[ofc_cpp] initial5", init_cards);
      }
    }
    for (int i=0;i<13;++i) boards_all.push_back(st.board[i]);
    rounds_all.push_back(static_cast<int16_t>(st.round_idx));
    if (verbose) {
      py::gil_scoped_acquire gil;
      py::print("  stored_round", rounds_all.back());
    }
    for (int i=0;i<3;++i) draws_all.push_back(st.draw3[i]);
    deck_sizes_all.push_back(static_cast<int16_t>(st.deck.size()));

    auto board_np = py::array_t<int16_t>(py::array::ShapeContainer{13});
    auto b_ptr = board_np.mutable_data();
    for (int i=0;i<13;++i) b_ptr[i] = st.board[i];
    auto deck_np = py::array_t<int16_t>(py::array::ShapeContainer{(ssize_t)st.deck.size()});
    auto d_ptr = deck_np.mutable_data();
    for (ssize_t i=0;i<(ssize_t)st.deck.size();++i) d_ptr[i] = st.deck[i];
    int action_count = 0;
    int action_idx = -1;

    if (st.round_idx == 0) {
      py::array_t<int16_t> placements = legal_actions_round0(board_np);
      action_count = static_cast<int>(placements.shape(0));

      if (action_count <= 0) {
        labels_all.push_back(0);
        action_counts_per_state.push_back(0);
        st.done = true;
        continue;
      }

      auto P = placements.unchecked<3>();
      py::array_t<int16_t> draw5(py::array::ShapeContainer{5});
      auto d5_ptr = draw5.mutable_data();
      for (int i=0;i<5;++i) d5_ptr[i] = st.initial5[i];

      for (int k=0;k<action_count;++k) {
        py::array_t<int16_t> slots_np(py::array::ShapeContainer{5});
        auto buf = slots_np.request();
        auto sl_ptr = static_cast<int16_t*>(buf.ptr);
        for (int i=0;i<5;++i) sl_ptr[i] = -1;
        for (int i=0;i<5;++i) {
          int card_idx = P(k,i,0);
          int slot_idx = P(k,i,1);
          if (card_idx >=0 && card_idx < 5) sl_ptr[card_idx] = slot_idx;
        }
        if (verbose && k==0) {
          py::gil_scoped_acquire gil;
          py::list slot_list;
          for (int i=0;i<5;++i) slot_list.append(sl_ptr[i]);
          py::print("[ofc_cpp] action", k, "slots_np", slot_list);
        }
        auto res = step_state_round0(board_np, draw5, deck_np, slots_np);
        auto next_board = res[0].cast<py::array_t<int16_t>>();
        auto next_round = res[1].cast<int>();
        auto next_draw = res[2].cast<py::array_t<int16_t>>();
        auto next_deck = res[3].cast<py::array_t<int16_t>>();
        record_action_state(next_board, next_round, next_draw, next_deck);
      }

      action_idx = sfl_choose_action(board_np, st.round_idx, draw5, deck_np);
      if (action_idx < 0 || action_idx >= action_count) action_idx = 0;

      py::array_t<int16_t> slots_np(py::array::ShapeContainer{5});
      auto buf = slots_np.request();
      auto sl_ptr = static_cast<int16_t*>(buf.ptr);
      for (int i=0;i<5;++i) sl_ptr[i] = -1;
      for (int i=0;i<5;++i) {
        int card_idx = P(action_idx,i,0);
        int slot_idx = P(action_idx,i,1);
        if (card_idx >=0 && card_idx < 5) sl_ptr[card_idx] = slot_idx;
      }
      auto res = step_state_round0(board_np, draw5, deck_np, slots_np);
      auto b2 = res[0].cast<py::array_t<int16_t>>();
      auto b2r = b2.unchecked<1>();
      for (int i=0;i<13;++i) st.board[i] = b2r(i);
      st.round_idx = res[1].cast<int>();
      auto draw_next = res[2].cast<py::array_t<int16_t>>();
      auto dr = draw_next.unchecked<1>();
      for (int i=0;i<3;++i) st.draw3[i] = dr(i);
      auto deck_next = res[3].cast<py::array_t<int16_t>>();
      auto dn = deck_next.unchecked<1>();
      st.deck.resize(dn.shape(0));
      for (ssize_t i=0;i<dn.shape(0);++i) st.deck[i] = dn(i);
      st.done = res[4].cast<bool>();
    } else {
      py::tuple acts = legal_actions_rounds1to4(board_np, st.round_idx);
      auto keeps = acts[0].cast<py::array_t<int8_t>>();
      auto places = acts[1].cast<py::array_t<int16_t>>();
      action_count = static_cast<int>(keeps.shape(0));

      if (action_count <= 0) {
        labels_all.push_back(0);
        action_counts_per_state.push_back(0);
        st.done = true;
        continue;
      }

      auto K = keeps.unchecked<2>();
      auto P = places.unchecked<3>();
      py::array_t<int16_t> draw3_np(py::array::ShapeContainer{3});
      auto d3_ptr = draw3_np.mutable_data();
      for (int i=0;i<3;++i) d3_ptr[i] = st.draw3[i];

      for (int k=0;k<action_count;++k) {
        int keep_i = K(k,0);
        int keep_j = K(k,1);
        int p00 = P(k,0,0);
        int p01 = P(k,0,1);
        int p10 = P(k,1,0);
        int p11 = P(k,1,1);
        auto res = step_state(board_np, st.round_idx, draw3_np, deck_np, keep_i, keep_j, p00, p01, p10, p11);
        auto next_board = res[0].cast<py::array_t<int16_t>>();
        auto next_round = res[1].cast<int>();
        auto next_draw = res[2].cast<py::array_t<int16_t>>();
        auto next_deck = res[3].cast<py::array_t<int16_t>>();
        record_action_state(next_board, next_round, next_draw, next_deck);
      }

      action_idx = sfl_choose_action(board_np, st.round_idx, draw3_np, deck_np);
      if (action_idx < 0 || action_idx >= action_count) action_idx = 0;

      int keep_i = K(action_idx,0);
      int keep_j = K(action_idx,1);
      int p00 = P(action_idx,0,0);
      int p01 = P(action_idx,0,1);
      int p10 = P(action_idx,1,0);
      int p11 = P(action_idx,1,1);
      auto res = step_state(board_np, st.round_idx, draw3_np, deck_np, keep_i, keep_j, p00, p01, p10, p11);
      auto b2 = res[0].cast<py::array_t<int16_t>>();
      auto b2r = b2.unchecked<1>();
      for (int i=0;i<13;++i) st.board[i] = b2r(i);
      st.round_idx = res[1].cast<int>();
      auto draw_next = res[2].cast<py::array_t<int16_t>>();
      auto dr = draw_next.unchecked<1>();
      for (int i=0;i<3;++i) st.draw3[i] = dr(i);
      auto deck_next = res[3].cast<py::array_t<int16_t>>();
      auto dn = deck_next.unchecked<1>();
      st.deck.resize(dn.shape(0));
      for (ssize_t i=0;i<dn.shape(0);++i) st.deck[i] = dn(i);
      st.done = res[4].cast<bool>();
    }

    labels_all.push_back(action_idx);
    action_counts_per_state.push_back(action_count);

    if (verbose) {
      py::gil_scoped_acquire gil;
      auto elapsed = std::chrono::steady_clock::now() - state_start;
      auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
      py::print("[ofc_cpp] state", n+1, "/", num_examples,
                "round", st.round_idx,
                "actions", action_count,
                "time_ms", ms,
                "done", st.done,
                "chosen_idx", action_idx);
    }

    if (st.done) {
      reset_state(st, rng);
    }
  }

  py::array_t<int16_t> boards_arr({num_examples,13});
  std::memcpy(boards_arr.mutable_data(), boards_all.data(), boards_all.size()*sizeof(int16_t));
  py::array_t<int16_t> rounds_arr(py::array::ShapeContainer{(ssize_t)num_examples});
  {
    auto Rdst = rounds_arr.mutable_data();
    for (int i=0;i<num_examples;++i) {
      int v = static_cast<int>(rounds_all[i]);
      Rdst[i] = static_cast<int16_t>(v);
    }
  }
  py::array_t<int16_t> draws_arr({num_examples,3});
  std::memcpy(draws_arr.mutable_data(), draws_all.data(), draws_all.size()*sizeof(int16_t));
  py::array_t<int16_t> deck_sizes_arr({num_examples});
  std::memcpy(deck_sizes_arr.mutable_data(), deck_sizes_all.data(), deck_sizes_all.size()*sizeof(int16_t));

  py::array_t<float> encoded = encode_state_batch_ints(boards_arr, rounds_arr, draws_arr, deck_sizes_arr);
  py::array_t<int32_t> labels_arr({num_examples});
  std::memcpy(labels_arr.mutable_data(), labels_all.data(), labels_all.size()*sizeof(int32_t));

  std::vector<int32_t> action_offsets(num_examples + 1, 0);
  for (size_t i = 0; i < action_counts_per_state.size(); ++i) {
    action_offsets[i + 1] = action_offsets[i] + action_counts_per_state[i];
  }
  int total_actions = action_offsets.back();
  py::array_t<int16_t> action_boards_arr({total_actions,13});
  std::memcpy(action_boards_arr.mutable_data(), action_boards.data(), action_boards.size()*sizeof(int16_t));
  py::array_t<int16_t> action_rounds_arr(py::array::ShapeContainer{(ssize_t)total_actions});
  std::memcpy(action_rounds_arr.mutable_data(), action_rounds.data(), action_rounds.size()*sizeof(int16_t));
  py::array_t<int16_t> action_draws_arr({total_actions,3});
  std::memcpy(action_draws_arr.mutable_data(), action_draws.data(), action_draws.size()*sizeof(int16_t));
  py::array_t<int16_t> action_deck_sizes_arr({total_actions});
  std::memcpy(action_deck_sizes_arr.mutable_data(), action_deck_sizes.data(), action_deck_sizes.size()*sizeof(int16_t));

  py::array_t<float> action_encoded = encode_state_batch_ints(action_boards_arr,
                                                              action_rounds_arr,
                                                              action_draws_arr,
                                                              action_deck_sizes_arr);

  py::list offsets_py;
  for (auto v : action_offsets) offsets_py.append(v);

  return py::make_tuple(encoded, labels_arr, offsets_py, action_encoded);
}

py::dict simulate_sfl_stats(uint64_t seed, int num_episodes) {
  const bool fast_mode = []() {
    const char* env = std::getenv("OFC_SFL_FAST");
    return env && env[0] != '\0' && env[0] != '0';
  }();
  FastSamplesGuard fast_guard(fast_mode);

  std::mt19937_64 rng(seed);
  SimpleState st;
  reset_state(st, rng);

  struct Stats {
    int episodes = 0;
    int fouls = 0;
    int passes = 0;
    int royalty_boards = 0;
    double reward_sum = 0.0;
    double royalty_sum = 0.0;
  } stats;

  auto copy_board_array = [](const SimpleState& st, py::array_t<int16_t>& board_np) {
    auto ptr = board_np.mutable_data();
    for (int i = 0; i < 13; ++i) ptr[i] = st.board[i];
  };

  auto copy_deck_array = [](const SimpleState& st, py::array_t<int16_t>& deck_np) {
    auto ptr = deck_np.mutable_data();
    for (ssize_t i = 0; i < (ssize_t)st.deck.size(); ++i) ptr[i] = st.deck[i];
  };

  auto copy_draw3 = [](const SimpleState& st, py::array_t<int16_t>& draw_np) {
    auto ptr = draw_np.mutable_data();
    for (int i = 0; i < 3; ++i) ptr[i] = st.draw3[i];
  };

  while (stats.episodes < num_episodes) {
    while (!st.done) {
      py::array_t<int16_t> board_np(py::array::ShapeContainer{13});
      copy_board_array(st, board_np);
      py::array_t<int16_t> deck_np(py::array::ShapeContainer{(ssize_t)st.deck.size()});
      copy_deck_array(st, deck_np);

      if (st.round_idx == 0) {
        py::array_t<int16_t> draw5(py::array::ShapeContainer{5});
        auto d5_ptr = draw5.mutable_data();
        for (int i = 0; i < 5; ++i) d5_ptr[i] = st.initial5[i];

        py::array_t<int16_t> placements = legal_actions_round0(board_np);
        int action_count = static_cast<int>(placements.shape(0));
        if (action_count <= 0) {
          st.done = true;
          break;
        }
        int action_idx = sfl_choose_action(board_np, st.round_idx, draw5, deck_np);
        if (action_idx < 0 || action_idx >= action_count) action_idx = 0;

        auto P = placements.unchecked<3>();
        py::array_t<int16_t> slots_np(py::array::ShapeContainer{5});
        auto slots_ptr = slots_np.mutable_data();
        for (int i = 0; i < 5; ++i) slots_ptr[i] = -1;
        for (int i = 0; i < 5; ++i) {
          int card_idx = P(action_idx, i, 0);
          int slot_idx = P(action_idx, i, 1);
          if (card_idx >= 0 && card_idx < 5) slots_ptr[card_idx] = slot_idx;
        }
        auto res = step_state_round0(board_np, draw5, deck_np, slots_np);
        auto b2 = res[0].cast<py::array_t<int16_t>>();
        auto b2r = b2.unchecked<1>();
        for (int i = 0; i < 13; ++i) st.board[i] = b2r(i);
        st.round_idx = res[1].cast<int>();
        auto draw_next = res[2].cast<py::array_t<int16_t>>();
        auto dr = draw_next.unchecked<1>();
        for (int i = 0; i < 3; ++i) st.draw3[i] = dr(i);
        auto deck_next = res[3].cast<py::array_t<int16_t>>();
        auto dn = deck_next.unchecked<1>();
        st.deck.resize(dn.shape(0));
        for (ssize_t i = 0; i < dn.shape(0); ++i) st.deck[i] = dn(i);
        st.done = res[4].cast<bool>();
      } else {
        py::tuple acts = legal_actions_rounds1to4(board_np, st.round_idx);
        auto keeps = acts[0].cast<py::array_t<int8_t>>();
        auto places = acts[1].cast<py::array_t<int16_t>>();
        int action_count = static_cast<int>(keeps.shape(0));
        if (action_count <= 0) {
          st.done = true;
          break;
        }
        py::array_t<int16_t> draw3_np(py::array::ShapeContainer{3});
        copy_draw3(st, draw3_np);

        int action_idx = sfl_choose_action(board_np, st.round_idx, draw3_np, deck_np);
        if (action_idx < 0 || action_idx >= action_count) action_idx = 0;

        auto K = keeps.unchecked<2>();
        auto P = places.unchecked<3>();
        int keep_i = K(action_idx, 0);
        int keep_j = K(action_idx, 1);
        int p00 = P(action_idx, 0, 0);
        int p01 = P(action_idx, 0, 1);
        int p10 = P(action_idx, 1, 0);
        int p11 = P(action_idx, 1, 1);
        auto res = step_state(board_np, st.round_idx, draw3_np, deck_np,
                              keep_i, keep_j, p00, p01, p10, p11);
        auto b2 = res[0].cast<py::array_t<int16_t>>();
        auto b2r = b2.unchecked<1>();
        for (int i = 0; i < 13; ++i) st.board[i] = b2r(i);
        st.round_idx = res[1].cast<int>();
        auto draw_next = res[2].cast<py::array_t<int16_t>>();
        auto dr = draw_next.unchecked<1>();
        for (int i = 0; i < 3; ++i) st.draw3[i] = dr(i);
        auto deck_next = res[3].cast<py::array_t<int16_t>>();
        auto dn = deck_next.unchecked<1>();
        st.deck.resize(dn.shape(0));
        for (ssize_t i = 0; i < dn.shape(0); ++i) st.deck[i] = dn(i);
        st.done = res[4].cast<bool>();
      }
    }

    std::array<int16_t,5> bottom{};
    std::array<int16_t,5> middle{};
    std::array<int16_t,3> top{};
    bool incomplete = false;
    for (int i = 0; i < 5; ++i) {
      bottom[i] = st.board[i];
      middle[i] = st.board[5 + i];
      if (bottom[i] < 0 || middle[i] < 0) incomplete = true;
    }
    for (int i = 0; i < 3; ++i) {
      top[i] = st.board[10 + i];
      if (top[i] < 0) incomplete = true;
    }

    float reward = -static_cast<float>(ofc::rules::FOUL_PENALTY);
    bool foul = true;
    int royalties = 0;
    if (!incomplete) {
      auto res = canonical_score_completed_board(bottom, middle, top);
      reward = res.first;
      foul = res.second;
      if (!foul) {
        HandType bottom_type = classify_hand(bottom);
        HandType middle_type = classify_hand(middle);
        royalties = five_card_royalty(bottom_type, false) +
                    five_card_royalty(middle_type, true) +
                    std::max(top_pair_royalty(top), top_trips_royalty(top));
      }
    }

    if (foul || incomplete) {
      stats.fouls++;
    } else if (royalties == 0) {
      stats.passes++;
    } else {
      stats.royalty_boards++;
      stats.royalty_sum += royalties;
    }
    stats.reward_sum += reward;
    stats.episodes++;

    reset_state(st, rng);
  }

  double total = static_cast<double>(stats.episodes);
  double foul_rate = total > 0 ? static_cast<double>(stats.fouls) / total : 0.0;
  double pass_rate = total > 0 ? static_cast<double>(stats.passes) / total : 0.0;
  double royalty_rate = total > 0 ? static_cast<double>(stats.royalty_boards) / total : 0.0;
  double avg_reward = total > 0 ? stats.reward_sum / total : 0.0;
  double avg_royalty = stats.royalty_boards > 0 ? stats.royalty_sum / static_cast<double>(stats.royalty_boards) : 0.0;

  py::dict out;
  out["episodes"] = stats.episodes;
  out["fast_mode"] = fast_mode;
  out["fouls"] = stats.fouls;
  out["passes"] = stats.passes;
  out["royalty_boards"] = stats.royalty_boards;
  out["foul_rate"] = foul_rate;
  out["pass_rate"] = pass_rate;
  out["royalty_rate"] = royalty_rate;
  out["avg_reward"] = avg_reward;
  out["avg_royalty"] = avg_royalty;
  return out;
}


