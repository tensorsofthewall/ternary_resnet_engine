#pragma once

#include "q2q_tensor.hpp"

namespace q2q {

void q8_maxpool_3x3_s2_p1(const QTensorNHWC& x, int OH, int OW, QTensorNHWC& y);

} // namespace q2q
