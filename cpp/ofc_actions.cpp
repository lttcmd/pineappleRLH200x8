#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cstdint>

namespace py = pybind11;

// Caps (keep in sync with Python)
static constexpr int MAX_EMPTY_SLOTS_CONSIDERED = 8;
static constexpr int MAX_SLOT_COMBOS_PER_PAIR = 6;

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


