#pragma once

#include <pybind11/numpy.h>

int sfl_choose_action(pybind11::array_t<int16_t> board,
                      int round_idx,
                      pybind11::array_t<int16_t> current_draw,
                      pybind11::array_t<int16_t> deck);


