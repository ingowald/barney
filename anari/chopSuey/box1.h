
#pragma once

// anari
#include "anari/anari_cpp/ext/linalg.h"

namespace anari::math {

  struct box1
  {
    box1() = default;
    box1(float lo, float up) : lower(lo), upper(up) {}

    float lower, upper;

    float size() const
    { return upper-lower; }

    float center() const
    { return lower+size()*0.5f; }

    box1& extend(float p) {
      lower = fminf(lower, p);
      upper = fmaxf(upper, p);
      return *this;
    }

    box1& extend(const box1 &b) {
      return extend(b.lower).extend(b.upper);
    }
  };
} // namespace anari::math
