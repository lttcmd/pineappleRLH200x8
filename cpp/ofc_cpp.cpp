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
py::array_t<int16_t> legal_actions_round0(py::array_t<int16_t> board);
// Forward decl from ofc_step.cpp round0
py::tuple step_state_round0(py::array_t<int16_t> board,
                            py::array_t<int16_t> current5,
                            py::array_t<int16_t> deck,
                            py::array_t<int16_t> slots5);
// Forward decl: encoding
py::array_t<float> encode_state_batch_ints(py::array_t<int16_t> boards,  // (N,13) -1 or 0..51
                                           py::array_t<int8_t> rounds,   // (N,)
                                           py::array_t<int16_t> draws,   // (N,3) -1 or 0..51
                                           py::array_t<int16_t> deck_sizes); // (N,)
// Episode generator
py::tuple generate_random_episodes(uint64_t seed, int num_episodes);
// Engine APIs
uint64_t create_engine(uint64_t seed);
void destroy_engine(uint64_t handle);
void engine_start_envs(uint64_t handle, int num_envs);
py::tuple request_policy_batch(uint64_t handle, int max_candidates_per_env);
int apply_policy_actions(uint64_t handle, py::array_t<int32_t> chosen);
py::tuple engine_collect_encoded_episodes(uint64_t handle);

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
  m.def("legal_actions_round0", &legal_actions_round0,
        py::arg("board"),
        "Generate round-0 placements; returns placements(n,5,2) of (card_idx, slot_idx).");
  m.def("step_state_round0", &step_state_round0,
        py::arg("board"), py::arg("current5"), py::arg("deck"), py::arg("slots5"),
        "Apply round-0 placement and deal next draw; returns (board, new_round, new_draw, new_deck, done).");
  m.def("encode_state_batch_ints", &encode_state_batch_ints,
        py::arg("boards"), py::arg("rounds"), py::arg("draws"), py::arg("deck_sizes"),
        "Encode states (ints) to feature matrix (N,838) float32.");
  m.def("generate_random_episodes", &generate_random_episodes,
        py::arg("seed"), py::arg("num_episodes"),
        "Generate random episodes end-to-end and return (encoded_states [S,838], episode_offsets [E+1], final_scores [E]).");

  // Engine bindings
  m.def("create_engine", &create_engine, py::arg("seed"),
        "Create a multi-env engine; returns handle (uint64).");
  m.def("destroy_engine", &destroy_engine, py::arg("handle"),
        "Destroy engine by handle.");
  m.def("engine_start_envs", &engine_start_envs, py::arg("handle"), py::arg("num_envs"),
        "Initialize N environments inside engine.");
  m.def("request_policy_batch", &request_policy_batch, py::arg("handle"), py::arg("max_candidates_per_env"),
        "Return (encoded_candidates [T,838], meta [T,3] int32: env_id, action_id, candidate_idx).");
  m.def("apply_policy_actions", &apply_policy_actions, py::arg("handle"), py::arg("chosen"),
        "Apply chosen actions array [N,2] int32 of (env_id, action_id). Returns number stepped.");
  m.def("engine_collect_encoded_episodes", &engine_collect_encoded_episodes, py::arg("handle"),
        "Collect accumulated selected encodings and scores; returns (encoded [S,838], offsets [E+1], scores [E]) and clears buffers.");
}


