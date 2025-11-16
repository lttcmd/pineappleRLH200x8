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
  // Copy inputs
  auto b_in = board.unchecked<1>();
  auto d_in = deck.unchecked<1>();
  auto cd_in = current_draw.unchecked<1>();

  std::array<int16_t,13> b{};
  for (int i=0;i<13;++i) b[i] = b_in(i);

  std::vector<int16_t> d(d_in.shape(0));
  for (ssize_t i=0;i<d_in.shape(0);++i) d[i] = d_in(i);

  std::array<int16_t,3> draw{};
  for (int i=0;i<3;++i) draw[i] = cd_in(i);

  int new_round = round_idx;

  if (round_idx == 0) {
    // Round 0 handling is provided by a dedicated API due to 5-card placement
    // The caller should use step_state_round0; here, fall back to no-op
    auto out_b = py::array_t<int16_t>({13});
    auto out_b_m = out_b.mutable_unchecked<1>();
    for (int i=0;i<13;++i) out_b_m(i) = b[i];
    auto out_draw = py::array_t<int16_t>({3});
    auto out_draw_m = out_draw.mutable_unchecked<1>();
    for (int i=0;i<3;++i) out_draw_m(i) = draw[i];
    auto out_deck = py::array_t<int16_t>({(ssize_t)d.size()});
    auto out_deck_m = out_deck.mutable_unchecked<1>();
    for (ssize_t i=0;i<(ssize_t)d.size();++i) out_deck_m(i) = d[i];
    bool done = false;
    return py::make_tuple(out_b, new_round, out_draw, out_deck, done);
  } else {
    // Keep two cards from current draw (indices 0..2)
    std::array<int16_t,2> kept{};
    kept[0] = (keep_i>=0 && keep_i<3) ? draw[keep_i] : int16_t(-1);
    kept[1] = (keep_j>=0 && keep_j<3) ? draw[keep_j] : int16_t(-1);

    // Place according to placements
    // placeX_keep_idx is 0 or 1 (which of kept), placeX_slot is 0..12
    // Ensure we use the correct kept card index
    if (place0_slot>=0 && place0_slot<13 && b[place0_slot] == -1 && place0_keep_idx>=0 && place0_keep_idx<2) {
      b[place0_slot] = kept[place0_keep_idx];
    }
    if (place1_slot>=0 && place1_slot<13 && b[place1_slot] == -1 && place1_keep_idx>=0 && place1_keep_idx<2) {
      b[place1_slot] = kept[place1_keep_idx];
    }

    // Advance round
    new_round = round_idx + 1;

    // Deal next 3 cards if not done
    std::array<int16_t,3> next_draw{ -1, -1, -1 };
    if (new_round <= 4) {
      if ((int)d.size() >= 3) {
        // Python used pop() from end 3 times
        for (int k=0;k<3;++k) {
          next_draw[k] = d.back();
          d.pop_back();
        }
      }
    }
    draw = next_draw;

    // Done check
    bool all_filled = true;
    for (int i=0;i<13;++i) if (b[i] == -1) { all_filled = false; break; }
    bool done = (new_round > 4) || all_filled;

    // Build outputs
    auto out_b = py::array_t<int16_t>({13});
    auto out_b_m = out_b.mutable_unchecked<1>();
    for (int i=0;i<13;++i) out_b_m(i) = b[i];
    auto out_draw = py::array_t<int16_t>({3});
    auto out_draw_m = out_draw.mutable_unchecked<1>();
    for (int i=0;i<3;++i) out_draw_m(i) = draw[i];
    auto out_deck = py::array_t<int16_t>({(ssize_t)d.size()});
    auto out_deck_m = out_deck.mutable_unchecked<1>();
    for (ssize_t i=0;i<(ssize_t)d.size();++i) out_deck_m(i) = d[i];

    return py::make_tuple(out_b, new_round, out_draw, out_deck, done);
  }
}

// Round 0 specialized step: place 5 cards and deal next draw
py::tuple step_state_round0(py::array_t<int16_t> board,
                            py::array_t<int16_t> current5,
                            py::array_t<int16_t> deck,
                            py::array_t<int16_t> slots5) {
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
    if (slot_idx>=0 && slot_idx<13) {
      b[slot_idx] = static_cast<int16_t>(card_value);
    }
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

  auto out_b = py::array_t<int16_t>({13});
  auto out_b_m = out_b.mutable_unchecked<1>();
  for (int i=0;i<13;++i) out_b_m(i) = b[i];
  auto out_draw = py::array_t<int16_t>({3});
  auto out_draw_m = out_draw.mutable_unchecked<1>();
  for (int i=0;i<3;++i) out_draw_m(i) = next_draw[i];
  auto out_deck = py::array_t<int16_t>({(ssize_t)d.size()});
  auto out_deck_m = out_deck.mutable_unchecked<1>();
  for (ssize_t i=0;i<(ssize_t)d.size();++i) out_deck_m(i) = d[i];
  return py::make_tuple(out_b, new_round, out_draw, out_deck, done);
}


