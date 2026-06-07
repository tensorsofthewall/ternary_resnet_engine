#include "q2q_tensor.hpp"

namespace q2q {

QTensorNHWC make_q8_tensor(int H_logical, int W_logical, int C, int border, float scale) {
    QTensorNHWC t;
    t.H = H_logical + 2 * border; t.W = W_logical + 2 * border; t.C = C;
    t.border = border; t.scale = scale;
    t.data.assign((size_t)t.H * t.W * C, uint8_t(0));
    return t;
}

} // namespace q2q
