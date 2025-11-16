#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <array>
#include <random>
#include <algorithm>
#include <cstdint>

namespace py = pybind11;

// Reuse helpers
py::array_t<int16_t> legal_actions_round0(py::array_t<int16_t> board);
py::tuple step_state_round0(py::array_t<int16_t> board,
                            py::array_t<int16_t> current5,
                            py::array_t<int16_t> deck,
                            py::array_t<int16_t> slots5);
py::tuple legal_actions_rounds1to4(py::array_t<int16_t> board, int round_idx);
py::tuple step_state(py::array_t<int16_t> board,
                     int round_idx,
                     py::array_t<int16_t> current_draw,
                     py::array_t<int16_t> deck,
                     int keep_i, int keep_j,
                     int place0_keep_idx, int place0_slot,
                     int place1_keep_idx, int place1_slot);
std::pair<float,bool> score_board_from_ints(py::array_t<int16_t> bottom,
                                            py::array_t<int16_t> middle,
                                            py::array_t<int16_t> top);
py::array_t<float> encode_state_batch_ints(py::array_t<int16_t> boards,
                                           py::array_t<int8_t> rounds,
                                           py::array_t<int16_t> draws,
                                           py::array_t<int16_t> deck_sizes);

static void init_deck(std::vector<int16_t>& deck) {
  deck.resize(52);
  for (int i=0;i<52;++i) deck[i] = static_cast<int16_t>(i);
}

// Generate random episodes end-to-end and return encoded states
// Returns (encoded_states [S,838], episode_offsets [E+1], final_scores [E])
py::tuple generate_random_episodes(uint64_t seed, int num_episodes) {
  std::mt19937_64 rng(seed);
  // Storage for all states
  std::vector<int16_t> boards_all; boards_all.reserve(num_episodes * 13 * 16);
  std::vector<int8_t> rounds_all; rounds_all.reserve(num_episodes * 16);
  std::vector<int16_t> draws_all; draws_all.reserve(num_episodes * 3 * 16);
  std::vector<int16_t> deck_sizes_all; deck_sizes_all.reserve(num_episodes * 16);
  std::vector<int32_t> offsets; offsets.reserve(num_episodes+1);
  std::vector<float> final_scores; final_scores.reserve(num_episodes);

  int32_t state_count = 0;
  offsets.push_back(state_count);

  for (int ep=0; ep<num_episodes; ++ep) {
    // init state
    std::vector<int16_t> deck;
    init_deck(deck);
    std::shuffle(deck.begin(), deck.end(), rng);
    std::array<int16_t,13> board; board.fill(-1);
    std::array<int16_t,3> draw3{ -1, -1, -1 };
    // initial 5 cards (top of deck is back)
    std::array<int16_t,5> initial5{};
    for (int i=0;i<5;++i){ initial5[i] = deck.back(); deck.pop_back(); }
    // legal actions round 0
    {
      auto b_np = py::array_t<int16_t>({13});
      auto b_m = b_np.mutable_unchecked<1>();
      for (int i=0;i<13;++i) b_m(i)=board[i];
      py::array_t<int16_t> placements = legal_actions_round0(b_np);
      if (placements.shape(0) == 0) { // fallback: skip episode
        final_scores.push_back(0.0f);
        offsets.push_back(state_count);
        continue;
      }
      std::uniform_int_distribution<int> uni0(0, placements.shape(0)-1);
      int pick = uni0(rng);
      // slots array from placements
      auto P = placements.unchecked<3>();
      std::array<int16_t,5> slots{};
      for (int i=0;i<5;++i){
        slots[i] = P(pick,i,1);
      }
      // step round 0
      auto c5 = py::array_t<int16_t>({5});
      auto c5m = c5.mutable_unchecked<1>();
      for (int i=0;i<5;++i) c5m(i)=initial5[i];
      auto d_np = py::array_t<int16_t>({(ssize_t)deck.size()});
      auto d_m = d_np.mutable_unchecked<1>();
      for (ssize_t i=0;i<(ssize_t)deck.size();++i) d_m(i) = deck[i];
      auto sl_np = py::array_t<int16_t>({5});
      auto sl_m = sl_np.mutable_unchecked<1>();
      for (int i=0;i<5;++i) sl_m(i)=slots[i];
      py::tuple res = step_state_round0(b_np, c5, d_np, sl_np);
      auto b2 = res[0].cast<py::array_t<int16_t>>();
      int new_round = res[1].cast<int>();
      auto d2 = res[3].cast<py::array_t<int16_t>>();
      auto draw_next = res[2].cast<py::array_t<int16_t>>();
      auto b2r = b2.unchecked<1>();
      for (int i=0;i<13;++i) board[i]=b2r(i);
      auto d2r = d2.unchecked<1>();
      deck.resize(d2r.shape(0));
      for (ssize_t i=0;i<d2r.shape(0);++i) deck[i]=d2r(i);
      auto dr = draw_next.unchecked<1>();
      for (int i=0;i<3;++i) draw3[i]=dr(i);
      // record state (after placements)
      for (int i=0;i<13;++i) boards_all.push_back(board[i]);
      rounds_all.push_back((int8_t)new_round);
      for (int i=0;i<3;++i) draws_all.push_back(draw3[i]);
      deck_sizes_all.push_back((int16_t)deck.size());
      state_count++;
    }
    // rounds 1..4
    int round_idx = 1;
    for (; round_idx <= 4; ++round_idx) {
      // legal actions
      auto b_np = py::array_t<int16_t>({13});
      auto b_m = b_np.mutable_unchecked<1>();
      for (int i=0;i<13;++i) b_m(i)=board[i];
      py::tuple acts = legal_actions_rounds1to4(b_np, round_idx);
      auto keeps = acts[0].cast<py::array_t<int8_t>>();
      auto places = acts[1].cast<py::array_t<int16_t>>();
      if (keeps.shape(0) == 0) break;
      std::uniform_int_distribution<int> uni(0, keeps.shape(0)-1);
      int pick = uni(rng);
      auto K = keeps.unchecked<2>();
      auto P = places.unchecked<3>();
      int keep_i = K(pick,0), keep_j=K(pick,1);
      int p00=P(pick,0,0), p01=P(pick,0,1);
      int p10=P(pick,1,0), p11=P(pick,1,1);
      // step
      auto d_np = py::array_t<int16_t>({(ssize_t)deck.size()});
      auto d_m = d_np.mutable_unchecked<1>();
      for (ssize_t i=0;i<(ssize_t)deck.size();++i) d_m(i) = deck[i];
      auto draw_np = py::array_t<int16_t>({3});
      auto draw_m = draw_np.mutable_unchecked<1>();
      for (int i=0;i<3;++i) draw_m(i)=draw3[i];
      py::tuple res = step_state(b_np, round_idx, draw_np, d_np, keep_i, keep_j, p00, p01, p10, p11);
      auto b2 = res[0].cast<py::array_t<int16_t>>();
      int new_round = res[1].cast<int>();
      auto draw_next = res[2].cast<py::array_t<int16_t>>();
      auto d2 = res[3].cast<py::array_t<int16_t>>();
      bool done = res[4].cast<bool>();
      auto b2r = b2.unchecked<1>();
      for (int i=0;i<13;++i) board[i]=b2r(i);
      auto d2r = d2.unchecked<1>();
      deck.resize(d2r.shape(0));
      for (ssize_t i=0;i<(ssize_t)deck.size();++i) deck[i]=d2r(i);
      auto dr = draw_next.unchecked<1>();
      for (int i=0;i<3;++i) draw3[i]=dr(i);
      // record state
      for (int i=0;i<13;++i) boards_all.push_back(board[i]);
      rounds_all.push_back((int8_t)new_round);
      for (int i=0;i<3;++i) draws_all.push_back(draw3[i]);
      deck_sizes_all.push_back((int16_t)deck.size());
      state_count++;
      if (done) break;
    }
    // score final board
    py::array_t<int16_t> bottom({5}), middle({5}), top({3});
    auto br = bottom.mutable_unchecked<1>();
    auto mr = middle.mutable_unchecked<1>();
    auto tr = top.mutable_unchecked<1>();
    for (int i=0;i<5;++i) br(i)=board[i];
    for (int i=0;i<5;++i) mr(i)=board[5+i];
    for (int i=0;i<3;++i) tr(i)=board[10+i];
    auto sc = score_board_from_ints(bottom, middle, top);
    final_scores.push_back(sc.first);
    offsets.push_back(state_count);
  }

  // Build numpy arrays
  ssize_t S = boards_all.size() / 13;
  py::array_t<int16_t> boards_np({S, (ssize_t)13});
  py::array_t<int8_t> rounds_np({S});
  py::array_t<int16_t> draws_np({S, (ssize_t)3});
  py::array_t<int16_t> deck_np({S});
  {
    auto B = boards_np.mutable_unchecked<2>();
    auto R = rounds_np.mutable_unchecked<1>();
    auto D = draws_np.mutable_unchecked<2>();
    auto L = deck_np.mutable_unchecked<1>();
    ssize_t idxB=0, idxD=0;
    for (ssize_t s=0;s<S;++s) {
      for (int i=0;i<13;++i) B(s,i) = boards_all[idxB++];
      R(s) = rounds_all[s];
      for (int i=0;i<3;++i) D(s,i) = draws_all[idxD++];
      L(s) = deck_sizes_all[s];
    }
  }
  py::array_t<float> encoded = encode_state_batch_ints(boards_np, rounds_np, draws_np, deck_np);
  py::array_t<int32_t> offs({(ssize_t)offsets.size()});
  auto O = offs.mutable_unchecked<1>();
  for (ssize_t i=0;i<(ssize_t)offsets.size();++i) O(i)=offsets[i];
  py::array_t<float> scores({(ssize_t)final_scores.size()});
  auto Sarr = scores.mutable_unchecked<1>();
  for (ssize_t i=0;i<(ssize_t)final_scores.size();++i) Sarr(i)=final_scores[i];
  return py::make_tuple(encoded, offs, scores);
}


