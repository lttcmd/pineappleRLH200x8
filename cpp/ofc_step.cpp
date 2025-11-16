#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <array>
#include <cstdint>

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
    // Place all 5 initial cards according to placements indices 0..4
    // In Python Action uses permutations(range(5)) mapped by placements list
    // Here we assume keep_i/keep_j ignored for round 0; placements slots use keep idx 0..4
    // To keep parity with Python, placements should carry card_idx 0..4.
    // We place any indices present in place0_keep_idx/place1_keep_idx and expect caller to provide remaining via repeated calls? 
    // For compatibility, we require caller to pass a full mapping via board already set before? Given complexity, we mimic Python two placements only for rounds>0.
    // For round 0, we won't handle here; caller should keep Python path. Return input unchanged.
    // Signal done false.
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
    if (place0_slot>=0 && place0_slot<13 && b[place0_slot] == -1) {
      int ki = (place0_keep_idx==0)?0:1;
      b[place0_slot] = kept[ki];
    }
    if (place1_slot>=0 && place1_slot<13 && b[place1_slot] == -1) {
      int ki = (place1_keep_idx==0)?0:1;
      b[place1_slot] = kept[ki];
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


