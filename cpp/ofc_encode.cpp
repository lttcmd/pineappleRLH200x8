#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cstdint>
#include <algorithm>

namespace py = pybind11;

// Encoding layout:
// 0..675: 13 slots * 52 one-hots
// 676..680: round one-hot (0..4)
// 681: deck size / 52.0
// 682..837: draw 3 * 52 one-hots (contiguous blocks)

py::array_t<float> encode_state_batch_ints(py::array_t<int16_t> boards,  // (N,13)
                                           py::array_t<int8_t> rounds,   // (N,)
                                           py::array_t<int16_t> draws,   // (N,3)
                                           py::array_t<int16_t> deck_sizes) { // (N,)
  auto B = boards.unchecked<2>();
  auto R = rounds.unchecked<1>();
  auto D = draws.unchecked<2>();
  auto L = deck_sizes.unchecked<1>();
  ssize_t N = B.shape(0);
  if (B.shape(1)!=13 || D.shape(1)!=3 || R.shape(0)!=N || L.shape(0)!=N) {
    throw std::runtime_error("encode_state_batch_ints: invalid shapes");
  }
  auto out = py::array_t<float>({N, (ssize_t)838});
  auto O = out.mutable_unchecked<2>();
  for (ssize_t n=0;n<N;++n) {
    // zero row
    for (int k=0;k<838;++k) O(n,k)=0.0f;
    // board
    int offset = 0;
    for (int i=0;i<13;++i) {
      int16_t v = B(n,i);
      if (v>=0 && v<52) {
        O(n, offset + v) = 1.0f;
      }
      offset += 52;
    }
    // round one-hot
    int rv = (int)R(n);
    if (rv>=0 && rv<5) O(n, 676 + rv) = 1.0f;
    // deck ratio
    O(n, 681) = std::min<int>(std::max<int>(L(n),0),52) / 52.0f;
    // draw 3
    offset = 682;
    for (int i=0;i<3;++i) {
      int16_t dv = D(n,i);
      if (dv>=0 && dv<52) O(n, offset + dv) = 1.0f;
      offset += 52;
    }
  }
  return out;
}


