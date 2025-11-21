#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cstdint>
#include <array>

namespace py = pybind11;

// Caps (keep in sync with Python)
static constexpr int MAX_EMPTY_SLOTS_CONSIDERED = 8;
static constexpr int MAX_SLOT_COMBOS_PER_PAIR = 6;
static constexpr int MAX_INITIAL_PERMUTATIONS = 12;

// Return keeps (M,2) int8 and placements (M,2,2) int16
// Inputs:
//  - board: int16[13], -1 empty else 0..51
//  - round_idx: int, must be 1..4 for this generator
py::tuple legal_actions_rounds1to4(py::array_t<int16_t> board, int round_idx) {
  auto b = board.unchecked<1>();
  if (b.shape(0) != 13) throw std::runtime_error("board must have 13 entries");
  if (round_idx <= 0) {
    // Not handled here; leave to Python
    return py::make_tuple(
      py::array_t<int8_t>(py::array::ShapeContainer{0,2}),
      py::array_t<int16_t>(py::array::ShapeContainer{0,2,2})
    );
  }
  // Collect empty slots
  std::vector<int16_t> empty;
  empty.reserve(13);
  for (int i=0;i<13;++i) if (b(i) == -1) empty.push_back(static_cast<int16_t>(i));
  if (empty.size() < 2) {
    return py::make_tuple(
      py::array_t<int8_t>(py::array::ShapeContainer{0,2}),
      py::array_t<int16_t>(py::array::ShapeContainer{0,2,2})
    );
  }
  // Subsample empty slots if large
  if (empty.size() > MAX_EMPTY_SLOTS_CONSIDERED) {
    empty.resize(MAX_EMPTY_SLOTS_CONSIDERED);
  }
  // Predefined keep pairs
  static const int8_t keep_pairs[3][2] = { {0,1}, {0,2}, {1,2} };

  // Precompute slot pairs with cap
  struct Pair { int16_t a; int16_t b; };
  std::vector<Pair> slot_pairs;
  slot_pairs.reserve(MAX_SLOT_COMBOS_PER_PAIR);
  // We'll iterate for each keep pair separately, so we don't need global storage

  // First estimate total actions
  int per_keep = 0;
  {
    int cnt = 0;
    for (size_t i=0; i<empty.size() && cnt<MAX_SLOT_COMBOS_PER_PAIR; ++i) {
      for (size_t j=i+1; j<empty.size() && cnt<MAX_SLOT_COMBOS_PER_PAIR; ++j) {
        ++cnt; // both orderings will be added later
      }
    }
    per_keep = cnt * 2; // both orderings
  }
  int total = per_keep * 3;
  if (total <= 0) {
    return py::make_tuple(
      py::array_t<int8_t>(py::array::ShapeContainer{0,2}),
      py::array_t<int16_t>(py::array::ShapeContainer{0,2,2})
    );
  }

  auto keeps = py::array_t<int8_t>({total, 2});
  auto places = py::array_t<int16_t>({total, 2, 2});
  auto K = keeps.mutable_unchecked<2>();
  auto P = places.mutable_unchecked<3>();

  int idx = 0;
  for (int kp=0; kp<3; ++kp) {
    // rebuild slot_pairs each keep to stay simple
    slot_pairs.clear();
    int cnt = 0;
    for (size_t i=0; i<empty.size() && cnt<MAX_SLOT_COMBOS_PER_PAIR; ++i) {
      for (size_t j=i+1; j<empty.size() && cnt<MAX_SLOT_COMBOS_PER_PAIR; ++j) {
        slot_pairs.push_back({empty[i], empty[j]});
        ++cnt;
      }
    }
    for (const auto& sp : slot_pairs) {
      // ordering 0->sp.a, 1->sp.b
      K(idx,0) = keep_pairs[kp][0];
      K(idx,1) = keep_pairs[kp][1];
      P(idx,0,0) = 0; P(idx,0,1) = sp.a;
      P(idx,1,0) = 1; P(idx,1,1) = sp.b;
      ++idx;
      // ordering 1->sp.a, 0->sp.b
      K(idx,0) = keep_pairs[kp][0];
      K(idx,1) = keep_pairs[kp][1];
      P(idx,0,0) = 1; P(idx,0,1) = sp.a;
      P(idx,1,0) = 0; P(idx,1,1) = sp.b;
      ++idx;
    }
  }

  return py::make_tuple(keeps, places);
}

// Round 0 actions: return (placements(n,5,2)) where each row is (card_idx, slot_idx)
py::array_t<int16_t> legal_actions_round0(py::array_t<int16_t> board) {
  auto b = board.unchecked<1>();
  if (b.shape(0) != 13) throw std::runtime_error("board must have 13 entries");
  // Collect empty slots
  std::vector<int16_t> empty;
  empty.reserve(13);
  static const int preferred_order[13] = {0,5,10,1,6,11,2,7,12,3,8,4,9};
  for (int idx : preferred_order) {
    if (b(idx) == -1) {
      empty.push_back(static_cast<int16_t>(idx));
    }
  }
  if (empty.size() < 5) {
    return py::array_t<int16_t>(py::array::ShapeContainer{0,5,2});
  }
  // Build up to MAX_INITIAL_PERMUTATIONS distinct slot selections.
  // We always try to include the "all on bottom" pattern first if the
  // 5 bottom slots are empty, so SFL can choose to lock a monster
  // (straight/flush/full-house/quads) on the bottom row when available.
  std::vector<std::array<int16_t,5>> slot_sets;
  slot_sets.reserve(MAX_INITIAL_PERMUTATIONS);

  // Special-case: all 5 bottom slots available
  {
    std::array<int16_t,5> bottom_slots{};
    int bc = 0;
    for (int s = 0; s < 5; ++s) {
      if (b(s) == -1) {
        bottom_slots[bc++] = static_cast<int16_t>(s);
      }
    }
    if (bc == 5) {
      slot_sets.push_back(bottom_slots);
    }
  }

  const size_t n = empty.size();
  for (size_t i0=0; i0<n && slot_sets.size()<MAX_INITIAL_PERMUTATIONS; ++i0) {
    for (size_t i1=i0+1; i1<n && slot_sets.size()<MAX_INITIAL_PERMUTATIONS; ++i1) {
      for (size_t i2=i1+1; i2<n && slot_sets.size()<MAX_INITIAL_PERMUTATIONS; ++i2) {
        for (size_t i3=i2+1; i3<n && slot_sets.size()<MAX_INITIAL_PERMUTATIONS; ++i3) {
          for (size_t i4=i3+1; i4<n && slot_sets.size()<MAX_INITIAL_PERMUTATIONS; ++i4) {
            slot_sets.push_back({empty[i0], empty[i1], empty[i2], empty[i3], empty[i4]});
          }
        }
      }
    }
  }
  if (slot_sets.empty()) {
    return py::array_t<int16_t>(py::array::ShapeContainer{0,5,2});
  }
  const int total = static_cast<int>(slot_sets.size());
  auto placements = py::array_t<int16_t>(py::array::ShapeContainer{total, 5, 2});
  auto P = placements.mutable_unchecked<3>();
  static const int perms[12][5] = {
    {0,1,2,3,4},{0,1,2,4,3},{0,1,3,2,4},{0,1,3,4,2},
    {0,2,1,3,4},{0,2,1,4,3},{0,2,3,1,4},{0,2,3,4,1},
    {1,0,2,3,4},{1,0,2,4,3},{1,0,3,2,4},{1,0,3,4,2}
  };
  for (int k=0;k<total;++k) {
    const auto& slots = slot_sets[k];
    const int* perm = perms[k % 12];
    for (int i=0;i<5;++i) {
      P(k,i,0) = static_cast<int16_t>(perm[i]);
      P(k,i,1) = slots[i];
    }
  }
  return placements;
}


