#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <utility>

namespace py = pybind11;

// Forward decl from ofc_score.cpp
std::pair<float, bool> score_board_from_ints(py::array_t<int16_t> bottom,
                                             py::array_t<int16_t> middle,
                                             py::array_t<int16_t> top);
// Forward decl from ofc_step.cpp
py::tuple step_state(py::array_t<int16_t> board,
                     int round_idx,
                     py::array_t<int16_t> current_draw,
                     py::array_t<int16_t> deck,
                     int keep_i, int keep_j,
                     int place0_keep_idx, int place0_slot,
                     int place1_keep_idx, int place1_slot);
// Forward decl from ofc_actions.cpp
py::tuple legal_actions_rounds1to4(py::array_t<int16_t> board, int round_idx);

PYBIND11_MODULE(ofc_cpp, m) {
  m.doc() = "C++ accelerated OFC helpers";
  m.def("score_board_from_ints", &score_board_from_ints,
        py::arg("bottom"), py::arg("middle"), py::arg("top"),
        "Score a complete board given 0..51 card integers for bottom(5), middle(5), top(3). "
        "Returns (score, fouled).");
  m.def("step_state", &step_state,
        py::arg("board"), py::arg("round_idx"), py::arg("current_draw"), py::arg("deck"),
        py::arg("keep_i"), py::arg("keep_j"),
        py::arg("place0_keep_idx"), py::arg("place0_slot"),
        py::arg("place1_keep_idx"), py::arg("place1_slot"),
        "Apply action to state and return (board, new_round, new_draw, new_deck, done).");
  m.def("legal_actions_rounds1to4", &legal_actions_rounds1to4,
        py::arg("board"), py::arg("round_idx"),
        "Generate legal actions for rounds 1..4 with caps; returns (keeps(n,2), placements(n,2,2)).");
}


