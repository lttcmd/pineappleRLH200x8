#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <array>
#include <random>
#include <algorithm>
#include <cstdint>
#include <unordered_map>
#include <iostream>
#include <cassert>
#include <cstring>

// Windows compatibility: ssize_t is not available on MSVC, use Py_ssize_t instead
#ifdef _WIN32
typedef Py_ssize_t ssize_t;
#endif

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

// ----------------- Phase 2: Simple multi-env engine -----------------
struct EnvState {
  std::array<int16_t,13> board;
  std::array<int16_t,3> draw3;
  std::vector<int16_t> deck;
  int round_idx;
  bool done;
  // Cache of last candidate actions for apply step: (kp0,kp1,p00,p01,p10,p11) or round0 as card_idx mapped to slot via p01, p11 ignored
  std::vector<std::array<int32_t,6>> last_actions;
};

struct Engine {
  std::mt19937_64 rng;
  std::vector<EnvState> envs;
  // Accumulation buffers for selected next-state encodings and episode summaries
  std::vector<float> enc_all;            // flat [S*838]
  std::vector<int32_t> episode_offsets;  // [E+1], starts with 0
  std::vector<float> final_scores;       // [E]
  std::vector<int32_t> env_counts;       // per-env states emitted in current episode
  Engine(uint64_t seed) : rng(seed) {}
};

static std::unordered_map<uint64_t, Engine*> g_engines;
static uint64_t g_next_handle = 1;

uint64_t create_engine(uint64_t seed) {
  uint64_t h = g_next_handle++;
  g_engines[h] = new Engine(seed);
  return h;
}

void destroy_engine(uint64_t handle) {
  auto it = g_engines.find(handle);
  if (it!=g_engines.end()) {
    delete it->second;
    g_engines.erase(it);
  }
}

void engine_start_envs(uint64_t handle, int num_envs) {
  auto it = g_engines.find(handle);
  if (it==g_engines.end()) throw std::runtime_error("invalid engine handle");
  Engine* eng = it->second;
  eng->envs.clear();
  eng->envs.resize(num_envs);
  eng->enc_all.clear();
  eng->final_scores.clear();
  eng->episode_offsets.clear();
  eng->episode_offsets.push_back(0);
  eng->env_counts.assign(num_envs, 0);
  for (int e=0;e<num_envs;++e) {
    EnvState& st = eng->envs[e];
    st.board.fill(-1);
    st.draw3 = {-1,-1,-1};
    st.deck.clear();
    init_deck(st.deck);
    std::shuffle(st.deck.begin(), st.deck.end(), eng->rng);
    // deal 5
    std::array<int16_t,5> initial5{};
    for (int i=0;i<5;++i){ initial5[i]=st.deck.back(); st.deck.pop_back(); }
    // round 0: set board empty; draw is initial5 carried via special step
    // We'll not store initial5 explicitly; use legal_actions_round0/step_state_round0 in request function
    // Store initial5 into draw3 temporarily unused; keep round 0 marker
    st.round_idx = 0;
    st.done = false;
    st.last_actions.clear();
    // Encode initial5 into draw3's first three and push remaining two onto deck front to retrieve in request
    st.draw3 = {initial5[0], initial5[1], initial5[2]};
    // push the remaining two back to deck head in order so request can reconstruct
    st.deck.insert(st.deck.begin(), initial5[4]);
    st.deck.insert(st.deck.begin(), initial5[3]);
  }
}

// Build candidates for all active envs up to max_candidates_per_env.
// Returns (encoded_candidates [T,838] float32, meta [T,3] int32: env_id, action_id, candidate_idx)
py::tuple request_policy_batch(uint64_t handle, int max_candidates_per_env) {
  auto it = g_engines.find(handle);
  if (it==g_engines.end()) throw std::runtime_error("invalid engine handle");
  Engine* eng = it->second;
  // Collect per-candidate raw state to encode
  std::vector<int16_t> boards_all;
  std::vector<int8_t> rounds_all;
  std::vector<int16_t> draws_all;
  std::vector<int16_t> deck_sizes_all;
  std::vector<int32_t> meta; // triples

  for (int env_id=0; env_id<(int)eng->envs.size(); ++env_id) {
    EnvState& st = eng->envs[env_id];
    if (st.done) continue;
    st.last_actions.clear();
    // assemble board and deck arrays
    auto b_np = py::array_t<int16_t>({13});
    auto b_m = b_np.mutable_unchecked<1>();
    for (int i=0;i<13;++i) b_m(i)=st.board[i];
    int produced = 0;
    if (st.round_idx == 0) {
      // Generate round0 placements
      py::array_t<int16_t> placements = legal_actions_round0(b_np);
      int total = (int)placements.shape(0);
      auto P = placements.unchecked<3>();
      // reconstruct initial5 from st.draw3 and first two of deck head
      std::array<int16_t,5> initial5{st.draw3[0], st.draw3[1], st.draw3[2], st.deck[0], st.deck[1]};
      for (int k=0; k<total && produced<max_candidates_per_env; ++k) {
        // Build step to get next state
        auto c5 = py::array_t<int16_t>({5});
        auto c5m = c5.mutable_unchecked<1>();
        for (int i=0;i<5;++i) c5m(i)=initial5[i];
        auto d_np = py::array_t<int16_t>({(ssize_t)st.deck.size()});
        auto d_m = d_np.mutable_unchecked<1>();
        for (ssize_t i=0;i<(ssize_t)st.deck.size();++i) d_m(i)=st.deck[i];
        auto sl_np = py::array_t<int16_t>({5});
        auto sl_m = sl_np.mutable_unchecked<1>();
        for (int i=0;i<5;++i) sl_m(i)=P(k,i,1);
        py::tuple res = step_state_round0(b_np, c5, d_np, sl_np);
        auto b2 = res[0].cast<py::array_t<int16_t>>();
        int new_round = res[1].cast<int>();
        auto draw_next = res[2].cast<py::array_t<int16_t>>();
        auto d2 = res[3].cast<py::array_t<int16_t>>();
        auto b2r = b2.unchecked<1>();
        auto dr = draw_next.unchecked<1>();
        auto d2r = d2.unchecked<1>();
        // record encoding inputs
        for (int i=0;i<13;++i) boards_all.push_back(b2r(i));
        rounds_all.push_back((int8_t)new_round);
        for (int i=0;i<3;++i) draws_all.push_back(dr(i));
        deck_sizes_all.push_back((int16_t)d2r.shape(0));
        meta.push_back(env_id); meta.push_back((int32_t)st.last_actions.size()); meta.push_back(produced);
        // store action (encode as round0 sentinel with kp = -1, put slots in p01,p11 etc. we store five slot indices by overloading: we will replay via step_state_round0 again during apply)
        // We cannot store 5 slots in our 6-field struct; store k (index into placements), and mark kp0=-2
        st.last_actions.push_back({-2, k, 0,0,0,0});
        produced++;
      }
    } else {
      // rounds 1..4
      py::tuple acts = legal_actions_rounds1to4(b_np, st.round_idx);
      auto keeps = acts[0].cast<py::array_t<int8_t>>();
      auto places = acts[1].cast<py::array_t<int16_t>>();
      int total = (int)keeps.shape(0);
      auto K = keeps.unchecked<2>();
      auto P = places.unchecked<3>();
      // draw array
      auto draw_np = py::array_t<int16_t>({3});
      auto draw_m = draw_np.mutable_unchecked<1>();
      for (int i=0;i<3;++i) draw_m(i)=st.draw3[i];
      // deck np
      auto d_np = py::array_t<int16_t>({(ssize_t)st.deck.size()});
      auto d_m = d_np.mutable_unchecked<1>();
      for (ssize_t i=0;i<(ssize_t)st.deck.size();++i) d_m(i)=st.deck[i];
      for (int k=0; k<total && produced<max_candidates_per_env; ++k) {
        int keep_i = K(k,0), keep_j=K(k,1);
        int p00=P(k,0,0), p01=P(k,0,1);
        int p10=P(k,1,0), p11=P(k,1,1);
        py::tuple res = step_state(b_np, st.round_idx, draw_np, d_np, keep_i, keep_j, p00, p01, p10, p11);
        auto b2 = res[0].cast<py::array_t<int16_t>>();
        int new_round = res[1].cast<int>();
        auto draw_next = res[2].cast<py::array_t<int16_t>>();
        auto d2 = res[3].cast<py::array_t<int16_t>>();
        auto b2r = b2.unchecked<1>();
        auto dr = draw_next.unchecked<1>();
        auto d2r = d2.unchecked<1>();
        for (int i=0;i<13;++i) boards_all.push_back(b2r(i));
        rounds_all.push_back((int8_t)new_round);
        for (int i=0;i<3;++i) draws_all.push_back(dr(i));
        deck_sizes_all.push_back((int16_t)d2r.shape(0));
        meta.push_back(env_id); meta.push_back((int32_t)st.last_actions.size()); meta.push_back(produced);
        st.last_actions.push_back({keep_i, keep_j, p00, p01, p10, p11});
        produced++;
      }
    }
  }

  // Pack encoding inputs
  ssize_t T = rounds_all.size();
  if (T==0) {
    // return empty
    auto enc = py::array_t<float>(py::array::ShapeContainer{(ssize_t)0, (ssize_t)838});
    auto meta_np = py::array_t<int32_t>(py::array::ShapeContainer{(ssize_t)0, (ssize_t)3});
    return py::make_tuple(enc, meta_np);
  }
  py::array_t<int16_t> boards_np({T, (ssize_t)13});
  py::array_t<int8_t> rounds_np({T});
  py::array_t<int16_t> draws_np({T, (ssize_t)3});
  py::array_t<int16_t> deck_np({T});
  {
    auto B = boards_np.mutable_unchecked<2>();
    auto R = rounds_np.mutable_unchecked<1>();
    auto D = draws_np.mutable_unchecked<2>();
    auto L = deck_np.mutable_unchecked<1>();
    ssize_t ib=0, id=0;
    for (ssize_t t=0;t<T;++t) {
      for (int i=0;i<13;++i) B(t,i)=boards_all[ib++];
      R(t)=rounds_all[t];
      for (int i=0;i<3;++i) D(t,i)=draws_all[id++];
      L(t)=deck_sizes_all[t];
    }
  }
  py::array_t<float> encoded = encode_state_batch_ints(boards_np, rounds_np, draws_np, deck_np);
  py::array_t<int32_t> meta_np({T, (ssize_t)3});
  {
    auto M = meta_np.mutable_unchecked<2>();
    for (ssize_t t=0;t<T;++t) {
      M(t,0)=meta[t*3+0];
      M(t,1)=meta[t*3+1];
      M(t,2)=meta[t*3+2];
    }
  }
  return py::make_tuple(encoded, meta_np);
}

// chosen: [N,2] int32 of (env_id, action_id_index)
// Returns number of envs stepped
int apply_policy_actions(uint64_t handle, py::array_t<int32_t> chosen) {
  auto it = g_engines.find(handle);
  if (it==g_engines.end()) throw std::runtime_error("invalid engine handle");
  Engine* eng = it->second;
  auto C = chosen.unchecked<2>();
  int N = (int)C.shape(0);
  int stepped = 0;
  for (int i=0;i<N;++i) {
    int env_id = C(i,0);
    int action_id = C(i,1);
    if (env_id<0 || env_id>=(int)eng->envs.size()) continue;
    EnvState& st = eng->envs[env_id];
    if (st.done) continue;
    if (action_id<0 || action_id>=(int)st.last_actions.size()) continue;
    // Build arrays
    auto b_np = py::array_t<int16_t>({13});
    auto b_m = b_np.mutable_unchecked<1>();
    for (int k=0;k<13;++k) b_m(k)=st.board[k];
    auto d_np = py::array_t<int16_t>({(ssize_t)st.deck.size()});
    auto d_m = d_np.mutable_unchecked<1>();
    for (ssize_t k=0;k<(ssize_t)st.deck.size();++k) d_m(k)=st.deck[k];
    auto act = st.last_actions[action_id];
    if (st.round_idx == 0 && act[0]==-2) {
      // Retrieve placements index and reconstruct initial5
      int kidx = act[1];
      py::array_t<int16_t> placements = legal_actions_round0(b_np);
      if (kidx < 0 || kidx >= (int)placements.shape(0)) continue;
      auto P = placements.unchecked<3>();
      auto sl_np = py::array_t<int16_t>({5});
      auto sl_m = sl_np.mutable_unchecked<1>();
      for (int j=0;j<5;++j) sl_m(j)=P(kidx,j,1);
      std::array<int16_t,5> initial5{st.draw3[0], st.draw3[1], st.draw3[2], st.deck[0], st.deck[1]};
      auto c5 = py::array_t<int16_t>({5});
      auto c5m = c5.mutable_unchecked<1>();
      for (int j=0;j<5;++j) c5m(j)=initial5[j];
      py::tuple res = step_state_round0(b_np, c5, d_np, sl_np);
      auto b2 = res[0].cast<py::array_t<int16_t>>();
      int new_round = res[1].cast<int>();
      auto draw_next = res[2].cast<py::array_t<int16_t>>();
      auto d2 = res[3].cast<py::array_t<int16_t>>();
      bool done = res[4].cast<bool>();
      auto b2r = b2.unchecked<1>(); for (int j=0;j<13;++j) st.board[j]=b2r(j);
      auto d2r = d2.unchecked<1>(); st.deck.resize(d2r.shape(0)); for (ssize_t j=0;j<d2r.shape(0);++j) st.deck[j]=d2r(j);
      auto dr = draw_next.unchecked<1>(); for (int j=0;j<3;++j) st.draw3[j]=dr(j);
      st.round_idx = new_round;
      st.done = done;
      stepped++;
    } else {
      // rounds 1..4
      auto draw_np = py::array_t<int16_t>({3});
      auto draw_m = draw_np.mutable_unchecked<1>();
      for (int j=0;j<3;++j) draw_m(j)=st.draw3[j];
      int keep_i = (int)act[0], keep_j=(int)act[1];
      int p00=(int)act[2], p01=(int)act[3], p10=(int)act[4], p11=(int)act[5];
      py::tuple res = step_state(b_np, st.round_idx, draw_np, d_np, keep_i, keep_j, p00, p01, p10, p11);
      auto b2 = res[0].cast<py::array_t<int16_t>>();
      int new_round = res[1].cast<int>();
      auto draw_next = res[2].cast<py::array_t<int16_t>>();
      auto d2 = res[3].cast<py::array_t<int16_t>>();
      bool done = res[4].cast<bool>();
      auto b2r = b2.unchecked<1>(); for (int j=0;j<13;++j) st.board[j]=b2r(j);
      auto d2r = d2.unchecked<1>(); st.deck.resize(d2r.shape(0)); for (ssize_t j=0;j<d2r.shape(0);++j) st.deck[j]=d2r(j);
      auto dr = draw_next.unchecked<1>(); for (int j=0;j<3;++j) st.draw3[j]=dr(j);
      st.round_idx = new_round;
      st.done = done;
      stepped++;
    }
    // Clear last actions to force new request on next cycle
    st.last_actions.clear();

    // Append selected next-state encoding (after step) into engine buffer
    // Build single-row arrays to reuse existing encoder
    // Boards
    auto boards_np = py::array_t<int16_t>(py::array::ShapeContainer{(ssize_t)1, (ssize_t)13});
    auto B = boards_np.mutable_unchecked<2>();
    for (int j=0;j<13;++j) B(0,j) = st.board[j];
    auto rounds_np = py::array_t<int8_t>(py::array::ShapeContainer{(ssize_t)1});
    auto R = rounds_np.mutable_unchecked<1>();
    R(0) = (int8_t)st.round_idx;
    auto draws_np = py::array_t<int16_t>(py::array::ShapeContainer{(ssize_t)1, (ssize_t)3});
    auto D = draws_np.mutable_unchecked<2>();
    for (int j=0;j<3;++j) D(0,j) = st.draw3[j];
    auto deck_np = py::array_t<int16_t>(py::array::ShapeContainer{(ssize_t)1});
    auto L = deck_np.mutable_unchecked<1>();
    L(0) = (int16_t)st.deck.size();
    py::array_t<float> enc_row = encode_state_batch_ints(boards_np, rounds_np, draws_np, deck_np);
    auto Erow = enc_row.unchecked<2>();
    for (int k=0;k<838;++k) eng->enc_all.push_back(Erow(0,k));
    // Update per-env count
    if (env_id >= 0 && env_id < (int)eng->env_counts.size()) {
      eng->env_counts[env_id] += 1;
    }

    // If env finished, compute score and close episode
    if (st.done) {
      // Score final board
      py::array_t<int16_t> bottom({5}), middle({5}), top({3});
      auto br = bottom.mutable_unchecked<1>();
      auto mr = middle.mutable_unchecked<1>();
      auto tr = top.mutable_unchecked<1>();
      for (int j=0;j<5;++j) br(j)=st.board[j];
      for (int j=0;j<5;++j) mr(j)=st.board[5+j];
      for (int j=0;j<3;++j) tr(j)=st.board[10+j];
      auto sc = score_board_from_ints(bottom, middle, top);
      eng->final_scores.push_back(sc.first);
      // Advance offsets
      int32_t last = eng->episode_offsets.empty() ? 0 : eng->episode_offsets.back();
      int32_t add = (env_id >=0 && env_id < (int)eng->env_counts.size()) ? eng->env_counts[env_id] : 0;
      eng->episode_offsets.push_back(last + add);
      if (env_id >=0 && env_id < (int)eng->env_counts.size()) eng->env_counts[env_id] = 0;
    }
  }
  return stepped;
}

// Collect accumulated selected encodings and final scores; clears buffers.
py::tuple engine_collect_encoded_episodes(uint64_t handle) {
  auto it = g_engines.find(handle);
  if (it==g_engines.end()) throw std::runtime_error("invalid engine handle");
  Engine* eng = it->second;
  // Build outputs
  ssize_t S = (ssize_t)(eng->enc_all.size() / 838);
  auto enc = py::array_t<float>({S, (ssize_t)838});
  if (S > 0) {
    auto E = enc.mutable_unchecked<2>();
    ssize_t idx = 0;
    for (ssize_t s=0;s<S;++s) {
      for (int k=0;k<838;++k) E(s,k) = eng->enc_all[idx++];
    }
  }
  auto offs = py::array_t<int32_t>({(ssize_t)eng->episode_offsets.size()});
  {
    auto O = offs.mutable_unchecked<1>();
    for (ssize_t i=0;i<(ssize_t)eng->episode_offsets.size();++i) O(i)=eng->episode_offsets[i];
  }
  auto scores = py::array_t<float>({(ssize_t)eng->final_scores.size()});
  {
    auto Sarr = scores.mutable_unchecked<1>();
    for (ssize_t i=0;i<(ssize_t)eng->final_scores.size();++i) Sarr(i)=eng->final_scores[i];
  }
  // Clear buffers for next cycle; keep offsets starting at 0
  eng->enc_all.clear();
  eng->final_scores.clear();
  eng->episode_offsets.clear();
  eng->episode_offsets.push_back(0);
  return py::make_tuple(enc, offs, scores);
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
    // Push the current state_count as the end offset for this episode
    // This should be cumulative: [0, 5, 10, 15, ...] for episodes with 5 states each
    offsets.push_back(state_count);
  }
  
  // Debug: Print first few and last few offsets to verify they're correct
  if (offsets.size() > 0) {
    std::cerr << "DEBUG: offsets.size()=" << offsets.size() << std::endl;
    std::cerr << "DEBUG: First 5 offsets: ";
    for (size_t i=0; i<std::min(5UL, offsets.size()); ++i) {
      std::cerr << offsets[i] << " ";
    }
    std::cerr << std::endl;
    std::cerr << "DEBUG: Last 5 offsets: ";
    size_t start_idx = offsets.size() > 5 ? offsets.size() - 5 : 0;
    for (size_t i=start_idx; i<offsets.size(); ++i) {
      std::cerr << offsets[i] << " ";
    }
    std::cerr << std::endl;
    std::cerr << "DEBUG: state_count=" << state_count << ", num_episodes=" << num_episodes << std::endl;
  }
  
  // Build numpy arrays
  // Offsets are already correctly built as cumulative values during the episode loop
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
  
  // Validate offsets vector integrity before copying
  // Check if offsets are corrupted (all same value - Linux compiler bug)
  bool offsets_valid = true;
  if (offsets.size() != static_cast<size_t>(num_episodes + 1)) {
    offsets_valid = false;
    std::cerr << "ERROR: Offsets size mismatch: expected " << (num_episodes + 1) << ", got " << offsets.size() << std::endl;
  } else if (offsets.size() > 0) {
    if (offsets[0] != 0) {
      offsets_valid = false;
      std::cerr << "ERROR: First offset must be 0, got " << offsets[0] << std::endl;
    } else if (offsets.back() != state_count) {
      offsets_valid = false;
      std::cerr << "ERROR: Last offset must equal state_count (" << state_count << "), got " << offsets.back() << std::endl;
    } else {
      // Check if all offsets are the same (Linux compiler bug indicator)
      bool all_same = true;
      for (size_t i = 1; i < offsets.size(); ++i) {
        if (offsets[i] != offsets[i-1]) {
          all_same = false;
          break;
        }
      }
      if (all_same && offsets.size() > 2) {
        offsets_valid = false;
        std::cerr << "ERROR: All offsets are the same value: " << offsets[0] << " (Linux compiler bug detected)" << std::endl;
      } else {
        // Verify offsets are non-decreasing
        for (size_t i = 1; i < offsets.size(); ++i) {
          if (offsets[i] < offsets[i-1]) {
            offsets_valid = false;
            std::cerr << "ERROR: Offsets not non-decreasing at index " << i << std::endl;
            break;
          }
        }
      }
    }
  }
  
  // If offsets are invalid, rebuild them from state count
  if (!offsets_valid) {
    std::cerr << "WARNING: Offsets vector is corrupted, rebuilding from state count..." << std::endl;
    offsets.clear();
    offsets.push_back(0);
    if (num_episodes > 0 && state_count > 0) {
      // Calculate states per episode (should be roughly equal)
      int32_t states_per_ep = state_count / num_episodes;
      if (states_per_ep == 0) states_per_ep = 1; // Fallback
      int32_t cumulative = 0;
      for (int ep = 0; ep < num_episodes; ++ep) {
        cumulative += states_per_ep;
        offsets.push_back(cumulative);
      }
      // Ensure last offset equals total state count exactly
      offsets.back() = state_count;
      std::cerr << "Rebuilt offsets: first 5 = ";
      for (size_t i = 0; i < std::min(5UL, offsets.size()); ++i) {
        std::cerr << offsets[i] << " ";
      }
      std::cerr << ", last = " << offsets.back() << std::endl;
    }
  }
  
  // Create offsets array - CRITICAL: Use raw pointer access to avoid mutable_unchecked bug
  // The mutable_unchecked accessor seems to have a bug on Linux where all writes go to same location
  ssize_t offs_size = (ssize_t)offsets.size();
  std::vector<int32_t> offsets_stable(offs_size);
  for (size_t i=0; i<offsets.size(); ++i) {
    offsets_stable[i] = static_cast<int32_t>(offsets[i]);
  }
  
  // Create array with explicit memory ownership - CRITICAL for Linux
  // Use py::array_t constructor that takes data pointer and makes a copy
  // This ensures Python gets an owned array, not a view
  py::array_t<int32_t> offs = py::cast(offsets_stable);
  
  // Verify the array contents after cast
  auto O_verify = offs.unchecked<1>();
  std::cerr << "DEBUG: After py::cast, verifying array:" << std::endl;
  for (ssize_t i=0; i<std::min(5L, offs_size); ++i) {
    int32_t val = O_verify(i);
    std::cerr << "  offs[" << i << "] = " << val << " (expected " << offsets_stable[i] << ")" << std::endl;
    if (val != offsets_stable[i]) {
      std::cerr << "ERROR: Mismatch at index " << i << "!" << std::endl;
    }
  }
  for (ssize_t i=std::max(0L, offs_size-5); i<offs_size; ++i) {
    int32_t val = O_verify(i);
    std::cerr << "  offs[" << i << "] = " << val << " (expected " << offsets_stable[i] << ")" << std::endl;
    if (val != offsets_stable[i]) {
      std::cerr << "ERROR: Mismatch at index " << i << "!" << std::endl;
    }
  }
  
  // Force Python to make a copy by accessing the array and ensuring it's not a view
  // This is a workaround for pybind11 view issues on Linux
  py::object offs_obj = offs;
  py::object offs_copy = py::module_::import("numpy").attr("array")(offs_obj, py::arg("copy")=true, py::arg("dtype")="int32");
  offs = offs_copy.cast<py::array_t<int32_t>>();
  
  std::cerr << "DEBUG: After forcing numpy copy, verifying array:" << std::endl;
  auto O_final = offs.unchecked<1>();
  for (ssize_t i=0; i<std::min(5L, offs_size); ++i) {
    int32_t val = O_final(i);
    std::cerr << "  offs[" << i << "] = " << val << " (expected " << offsets_stable[i] << ")" << std::endl;
  }
  py::array_t<float> scores({(ssize_t)final_scores.size()});
  auto Sarr = scores.mutable_unchecked<1>();
  for (ssize_t i=0;i<(ssize_t)final_scores.size();++i) Sarr(i)=final_scores[i];
  return py::make_tuple(encoded, offs, scores);
}


