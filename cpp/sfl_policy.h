#pragma once

#include <pybind11/numpy.h>

int sfl_choose_action(pybind11::array_t<int16_t> board,
                      int round_idx,
                      pybind11::array_t<int16_t> current_draw,
                      pybind11::array_t<int16_t> deck);

// Configure SFL rollout reward shaping (planning-only, does not affect real scoring).
// All arguments are float32-like; typical signs:
//   foul_penalty  < 0   (e.g. -5.0)
//   pass_penalty  < 0   (e.g. -3.0)
//   medium_bonus  >= 0  (added for 4<=royalties<8)
//   strong_bonus  >= 0  (added for 8<=royalties<12)
//   monster_mult  >= 0  (multiplier applied to royalties when royalties>=12)
void set_sfl_shaping(float foul_penalty,
                     float pass_penalty,
                     float medium_bonus,
                     float strong_bonus,
                     float monster_mult);


