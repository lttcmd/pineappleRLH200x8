#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <array>
#include <cstdint>

// Windows compatibility: ssize_t is not available on MSVC, use Py_ssize_t instead
#ifdef _WIN32
typedef Py_ssize_t ssize_t;
#endif

namespace py = pybind11;

// step_state:
// Inputs:
// - board: int16[13], -1 if empty, else 0..51
// - round_idx: int (0..5), >4 means done
// - current_draw: int16[3], -1 if unused
// - deck: int16[N] (top at back as in Python code using pop())
// - keep_i, keep_j: indices of the two kept cards from current_draw (0..2) or 0..4 in round 0
// - place0_keep_idx, place0_slot, place1_keep_idx, place1_slot
//
// Behavior matches Python ofc_env.step simplified rules.
//
// Returns tuple(board[13], new_round, new_draw[3], new_deck[N], done)
py::tuple step_state(py::array_t<int16_t> board,
                     int round_idx,
                     py::array_t<int16_t> current_draw,
                     py::array_t<int16_t> deck,
                     int keep_i, int keep_j,
                     int place0_keep_idx, int place0_slot,
                     int place1_keep_idx, int place1_slot) {
  // Copy inputs safely into local containers
  auto b_in = board.unchecked<1>();
  auto d_in = deck.unchecked<1>();
  auto cd_in = current_draw.unchecked<1>();

  std::array<int16_t, 13> b;
  for (int i = 0; i < 13; ++i) {
    b[i] = b_in(i);
  }

  std::vector<int16_t> d(d_in.shape(0));
  for (ssize_t i = 0; i < d_in.shape(0); ++i) {
    d[i] = d_in(i);
  }

  std::array<int16_t, 3> draw;
  for (int i = 0; i < 3; ++i) {
    draw[i] = cd_in(i);
  }

  int new_round = round_idx;

  // Round 0 is handled by step_state_round0; keep this as a no-op for safety.
  if (round_idx <= 0) {
    auto out_b = py::array_t<int16_t>(py::array::ShapeContainer{13});
    auto out_b_m = out_b.mutable_unchecked<1>();
    for (int i = 0; i < 13; ++i) out_b_m(i) = b[i];
    auto out_draw = py::array_t<int16_t>(py::array::ShapeContainer{3});
    auto out_draw_m = out_draw.mutable_unchecked<1>();
    for (int i = 0; i < 3; ++i) out_draw_m(i) = draw[i];
    auto out_deck = py::array_t<int16_t>(py::array::ShapeContainer{static_cast<ssize_t>(d.size())});
    auto out_deck_m = out_deck.mutable_unchecked<1>();
    for (ssize_t i = 0; i < static_cast<ssize_t>(d.size()); ++i) out_deck_m(i) = d[i];
    bool done = false;
    return py::make_tuple(out_b, new_round, out_draw, out_deck, done);
  }

  // Keep two cards from the current 3-card draw.
  std::array<int16_t, 2> kept{int16_t(-1), int16_t(-1)};
  if (keep_i >= 0 && keep_i < 3) kept[0] = draw[keep_i];
  if (keep_j >= 0 && keep_j < 3) kept[1] = draw[keep_j];

  auto place_card = [&](int keep_idx, int slot_idx) {
    if (keep_idx < 0 || keep_idx > 1) return;
    if (slot_idx < 0 || slot_idx >= 13) return;
    if (kept[keep_idx] < 0) return;
    if (b[slot_idx] != -1) return;
    b[slot_idx] = kept[keep_idx];
  };

  place_card(place0_keep_idx, place0_slot);
  place_card(place1_keep_idx, place1_slot);

  // Advance round and deal next 3 cards, mirroring Python pop-from-end logic.
  new_round = round_idx + 1;
  std::array<int16_t, 3> next_draw{int16_t(-1), int16_t(-1), int16_t(-1)};
  if (new_round <= 4 && static_cast<int>(d.size()) >= 3) {
    for (int k = 0; k < 3; ++k) {
      next_draw[k] = d.back();
      d.pop_back();
    }
  }
  draw = next_draw;

  // Done when either all slots are filled or we have advanced past round 4.
  bool all_filled = true;
  for (int i = 0; i < 13; ++i) {
    if (b[i] == -1) {
      all_filled = false;
      break;
    }
  }
  bool done = (new_round > 4) || all_filled;

  // Build outputs
  auto out_b = py::array_t<int16_t>(py::array::ShapeContainer{13});
  auto out_b_m = out_b.mutable_unchecked<1>();
  for (int i = 0; i < 13; ++i) out_b_m(i) = b[i];

  auto out_draw = py::array_t<int16_t>(py::array::ShapeContainer{3});
  auto out_draw_m = out_draw.mutable_unchecked<1>();
  for (int i = 0; i < 3; ++i) out_draw_m(i) = draw[i];

  auto out_deck = py::array_t<int16_t>(py::array::ShapeContainer{static_cast<ssize_t>(d.size())});
  auto out_deck_m = out_deck.mutable_unchecked<1>();
  for (ssize_t i = 0; i < static_cast<ssize_t>(d.size()); ++i) out_deck_m(i) = d[i];

  return py::make_tuple(out_b, new_round, out_draw, out_deck, done);
}

// Round 0 specialized step: place 5 cards and deal next draw
py::tuple step_state_round0(py::array_t<int16_t> board,
                            py::array_t<int16_t> current5,
                            py::array_t<int16_t> deck,
                            py::array_t<int16_t> slots5) {
  const bool debug = []() {
    const char* env = std::getenv("OFC_STEP_DEBUG");
    return env && env[0] != '\0' && env[0] != '0';
  }();
  // Validate shapes
  if (board.shape(0) != 13) throw std::runtime_error("board must have 13 elements");
  if (current5.shape(0) != 5) throw std::runtime_error("current5 must have 5 elements");
  if (slots5.shape(0) != 5) throw std::runtime_error("slots5 must have 5 elements");
  
  auto b_in = board.unchecked<1>();
  auto d_in = deck.unchecked<1>();
  auto c5 = current5.unchecked<1>();
  auto sl = slots5.unchecked<1>(); // shape (5,)

  // Initialize board from input (all should be -1 for empty)
  std::array<int16_t,13> b;
  for (int i=0;i<13;++i) {
    b[i] = b_in(i);
  }
  std::vector<int16_t> d(d_in.shape(0));
  for (ssize_t i=0;i<d_in.shape(0);++i) d[i] = d_in(i);

  // Place all 5: card index i goes to slots5[i]
  // DEBUG: Test if output mechanism works with manual values
  // Uncomment below to test:
  // b[0] = 42; b[1] = 26; b[2] = 47; b[3] = 7; b[4] = 27;
  
  // Read values from input arrays
  for (int i=0;i<5;++i) {
    int slot_idx = static_cast<int>(sl(i));
    int card_value = static_cast<int>(c5(i));
    if (debug) {
      py::gil_scoped_acquire gil;
      py::print("[step_state_round0] card_index", i, "value", card_value, "slot", slot_idx);
    }
    if (slot_idx>=0 && slot_idx<13) {
      b[slot_idx] = static_cast<int16_t>(card_value);
    }
  }
  if (debug) {
    py::gil_scoped_acquire gil;
    py::list board_list;
    for (int i=0;i<13;++i) board_list.append(b[i]);
    py::print("[step_state_round0] board_after", board_list);
  }
  // Deal next 3 for round 1
  std::array<int16_t,3> next_draw{ -1, -1, -1 };
  if ((int)d.size() >= 3) {
    for (int k=0;k<3;++k) {
      next_draw[k] = d.back();
      d.pop_back();
    }
  }
  int new_round = 1;
  bool all_filled = true;
  for (int i=0;i<13;++i) if (b[i] == -1) { all_filled = false; break; }
  bool done = (new_round > 4) || all_filled;

  auto out_b = py::array_t<int16_t>(py::array::ShapeContainer{13});
  auto out_b_m = out_b.mutable_unchecked<1>();
  for (int i=0;i<13;++i) out_b_m(i) = b[i];
  auto out_draw = py::array_t<int16_t>(py::array::ShapeContainer{3});
  auto out_draw_m = out_draw.mutable_unchecked<1>();
  for (int i=0;i<3;++i) out_draw_m(i) = next_draw[i];
  auto out_deck = py::array_t<int16_t>(py::array::ShapeContainer{(ssize_t)d.size()});
  auto out_deck_m = out_deck.mutable_unchecked<1>();
  for (ssize_t i=0;i<(ssize_t)d.size();++i) out_deck_m(i) = d[i];
  return py::make_tuple(out_b, new_round, out_draw, out_deck, done);
}


