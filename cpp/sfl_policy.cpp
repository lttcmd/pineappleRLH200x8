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
                                           py::array_t<int8_t> rounds,
                                           py::array_t<int16_t> draws,
                                           py::array_t<int16_t> deck_sizes);

namespace {

constexpr std::array<int, 5> kSamplesPerRound = {48, 36, 24, 16, 8};
constexpr std::array<int, 5> kSamplesPerRoundFast = {2, 2, 1, 1, 1};

thread_local bool g_force_fast_samples = false;

struct FastSamplesGuard {
  bool prev;
  FastSamplesGuard(bool enable) : prev(g_force_fast_samples) {
    if (enable) g_force_fast_samples = true;
  }
  ~FastSamplesGuard() { g_force_fast_samples = prev; }
};

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

float score_completed_board(const std::array<int16_t, 5>& bottom,
                            const std::array<int16_t, 5>& middle,
                            const std::array<int16_t, 3>& top) {
  HandType bottom_type = classify_hand(bottom);
  HandType middle_type = classify_hand(middle);
  int bottom_rank = hand_rank_value(bottom_type);
  int middle_rank = hand_rank_value(middle_type);

  bool foul = bottom_rank <= middle_rank;
  std::array<int, 3> top_ranks{};
  for (int i = 0; i < 3; ++i) top_ranks[i] = rank_from_card(top[i]);
  int top_pair = top_pair_royalty(top);
  int top_trips = top_trips_royalty(top);

  if (!foul) {
    if (top_trips > 0) {
      switch (middle_type) {
        case HandType::kTrips:
        case HandType::kFullHouse:
        case HandType::kQuads:
        case HandType::kStraightFlush:
        case HandType::kRoyalFlush:
          break;
        default:
          foul = true;
          break;
      }
    } else if (top_pair > 0) {
      if (middle_rank < hand_rank_value(HandType::kTrips)) foul = true;
    } else {
      if (middle_type == HandType::kHighCard) {
        int mid_max = 0;
        for (int i = 0; i < 5; ++i) mid_max = std::max(mid_max, rank_from_card(middle[i]));
        int top_max = std::max({top_ranks[0], top_ranks[1], top_ranks[2]});
        if (mid_max <= top_max) foul = true;
      }
    }
  }

  if (foul) {
    return -static_cast<float>(ofc::rules::FOUL_PENALTY);
  }

  int royalties = five_card_royalty(bottom_type, false) +
                  five_card_royalty(middle_type, true) +
                  std::max(top_pair, top_trips);
  return static_cast<float>(royalties);
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
    return score_completed_board(bottom_full, middle_full, top_full);
  }

  if (static_cast<int>(deck.size()) < total_need) {
    return -static_cast<float>(ofc::rules::FOUL_PENALTY);
  }

  int clamped_round = std::clamp(next_round, 0, 4);
  const auto& table = g_force_fast_samples ? kSamplesPerRoundFast : kSamplesPerRound;
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
        float score = score_completed_board(bottom_full, middle_full, top_full);
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
    for (int idx = 0; idx < total; ++idx) {
      py::array_t<int16_t> slots_np({5});
      auto slots = slots_np.mutable_unchecked<1>();
      for (int i = 0; i < 5; ++i) slots(i) = placement_view(idx, i, 1);
      auto res = step_state_round0(board_np, current_draw, deck_np, slots_np);
      auto next_board = res[0].cast<py::array_t<int16_t>>();
      int next_round = res[1].cast<int>();
      auto next_deck = res[3].cast<py::array_t<int16_t>>();
      auto board_arr = board_from_numpy(next_board);
      auto deck_vec = deck_from_numpy(next_deck);
      float score = rollout_value(board_arr, next_round, deck_vec);
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
  const bool verbose = debug_mode || num_examples <= 16;

  std::mt19937_64 rng(seed);
  SimpleState st;
  reset_state(st, rng);

  std::vector<int16_t> boards_all;
  std::vector<int8_t> rounds_all;
  std::vector<int16_t> draws_all;
  std::vector<int16_t> deck_sizes_all;
  std::vector<int32_t> labels_all;

  std::vector<int16_t> action_boards;
  std::vector<int8_t> action_rounds;
  std::vector<int16_t> action_draws;
  std::vector<int16_t> action_deck_sizes;
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
    action_rounds.push_back(static_cast<int8_t>(next_round));
    auto d = draw_arr.unchecked<1>();
    for (int i=0;i<3;++i) {
      int16_t val = (i < d.shape(0)) ? d(i) : static_cast<int16_t>(-1);
      action_draws.push_back(val);
    }
    auto deck_view = deck_arr.unchecked<1>();
    action_deck_sizes.push_back(static_cast<int16_t>(deck_view.shape(0)));
  };

  for (int n=0;n<num_examples;++n) {
    auto state_start = std::chrono::steady_clock::now();
    for (int i=0;i<13;++i) boards_all.push_back(st.board[i]);
    rounds_all.push_back(static_cast<int8_t>(st.round_idx));
    for (int i=0;i<3;++i) draws_all.push_back(st.draw3[i]);
    deck_sizes_all.push_back(static_cast<int16_t>(st.deck.size()));

    auto board_np = py::array_t<int16_t>({13});
    auto b_m = board_np.mutable_unchecked<1>();
    for (int i=0;i<13;++i) b_m(i) = st.board[i];
    auto deck_np = py::array_t<int16_t>({(ssize_t)st.deck.size()});
    auto d_m = deck_np.mutable_unchecked<1>();
    for (ssize_t i=0;i<(ssize_t)st.deck.size();++i) d_m(i) = st.deck[i];
    py::array_t<int16_t> draw_np = (st.round_idx == 0)
        ? py::array_t<int16_t>({5})
        : py::array_t<int16_t>({3});
    if (st.round_idx == 0) {
      auto draw_m = draw_np.mutable_unchecked<1>();
      for (int i=0;i<5;++i) draw_m(i) = st.initial5[i];
    } else {
      auto draw_m = draw_np.mutable_unchecked<1>();
      for (int i=0;i<3;++i) draw_m(i) = st.draw3[i];
    }

    int action_idx = sfl_choose_action(board_np, st.round_idx, draw_np, deck_np);
    labels_all.push_back(action_idx);

    int action_count = 0;
    if (st.round_idx == 0) {
      py::array_t<int16_t> placements = legal_actions_round0(board_np);
      auto P = placements.unchecked<3>();
      action_count = static_cast<int>(placements.shape(0));

      for (int k=0;k<action_count;++k) {
        py::array_t<int16_t> board_copy({13});
        auto bc = board_copy.mutable_unchecked<1>();
        for (int i=0;i<13;++i) bc(i) = st.board[i];

        py::array_t<int16_t> deck_copy({(ssize_t)st.deck.size()});
        auto dc = deck_copy.mutable_unchecked<1>();
        for (ssize_t i=0;i<(ssize_t)st.deck.size();++i) dc(i) = st.deck[i];

        py::array_t<int16_t> draw_copy({5});
        auto drawc = draw_copy.mutable_unchecked<1>();
        for (int i=0;i<5;++i) drawc(i) = st.initial5[i];

        py::array_t<int16_t> slots_np({5});
        auto sl = slots_np.mutable_unchecked<1>();
        for (int i=0;i<5;++i) sl(i) = P(k,i,1);

        auto res = step_state_round0(board_copy, draw_copy, deck_copy, slots_np);
        auto next_board = res[0].cast<py::array_t<int16_t>>();
        auto next_round = res[1].cast<int>();
        auto next_draw = res[2].cast<py::array_t<int16_t>>();
        auto next_deck = res[3].cast<py::array_t<int16_t>>();
        record_action_state(next_board, next_round, next_draw, next_deck);
      }

      auto sl_np = py::array_t<int16_t>({5});
      auto sl_m = sl_np.mutable_unchecked<1>();
      for (int i=0;i<5;++i) sl_m(i) = P(action_idx, i, 1);
      auto c5_np = py::array_t<int16_t>({5});
      auto c5_m = c5_np.mutable_unchecked<1>();
      for (int i=0;i<5;++i) c5_m(i) = st.initial5[i];
      auto res = step_state_round0(board_np, c5_np, deck_np, sl_np);
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
      auto K = keeps.unchecked<2>();
      auto P = places.unchecked<3>();
      action_count = static_cast<int>(keeps.shape(0));

      for (int k=0;k<action_count;++k) {
        py::array_t<int16_t> board_copy({13});
        auto bc = board_copy.mutable_unchecked<1>();
        for (int i=0;i<13;++i) bc(i) = st.board[i];

        py::array_t<int16_t> deck_copy({(ssize_t)st.deck.size()});
        auto dc = deck_copy.mutable_unchecked<1>();
        for (ssize_t i=0;i<(ssize_t)st.deck.size();++i) dc(i) = st.deck[i];

        py::array_t<int16_t> draw_copy({3});
        auto drawc = draw_copy.mutable_unchecked<1>();
        for (int i=0;i<3;++i) drawc(i) = st.draw3[i];

        int keep_i = K(k,0);
        int keep_j = K(k,1);
        int p00 = P(k,0,0);
        int p01 = P(k,0,1);
        int p10 = P(k,1,0);
        int p11 = P(k,1,1);

        auto res = step_state(board_copy, st.round_idx, draw_copy, deck_copy, keep_i, keep_j, p00, p01, p10, p11);
        auto next_board = res[0].cast<py::array_t<int16_t>>();
        auto next_round = res[1].cast<int>();
        auto next_draw = res[2].cast<py::array_t<int16_t>>();
        auto next_deck = res[3].cast<py::array_t<int16_t>>();
        record_action_state(next_board, next_round, next_draw, next_deck);
      }

      int keep_i = K(action_idx,0);
      int keep_j = K(action_idx,1);
      int p00 = P(action_idx,0,0);
      int p01 = P(action_idx,0,1);
      int p10 = P(action_idx,1,0);
      int p11 = P(action_idx,1,1);
      auto res = step_state(board_np, st.round_idx, draw_np, deck_np, keep_i, keep_j, p00, p01, p10, p11);
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

    action_counts_per_state.push_back(action_count);

    if (verbose) {
      py::gil_scoped_acquire gil;
      auto elapsed = std::chrono::steady_clock::now() - state_start;
      auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
      py::print("[ofc_cpp] state", n+1, "/", num_examples,
                "round", st.round_idx,
                "actions", action_count,
                "time_ms", ms);
    }

    if (st.done) {
      reset_state(st, rng);
    }
  }

  py::array_t<int16_t> boards_arr({num_examples,13});
  std::memcpy(boards_arr.mutable_data(), boards_all.data(), boards_all.size()*sizeof(int16_t));
  py::array_t<int8_t> rounds_arr({num_examples});
  std::memcpy(rounds_arr.mutable_data(), rounds_all.data(), rounds_all.size()*sizeof(int8_t));
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
  py::array_t<int8_t> action_rounds_arr({total_actions});
  std::memcpy(action_rounds_arr.mutable_data(), action_rounds.data(), action_rounds.size()*sizeof(int8_t));
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


