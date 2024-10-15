
#pragma once

// anari
#include "anari/anari_cpp/ext/linalg.h"

namespace anari::math {

  struct box3
  {
    box3() = default;
    box3(anari::math::float3 lo, anari::math::float3 up) : lower(lo), upper(up) {}

    anari::math::float3 lower, upper;

    anari::math::float3 size() const
    { return upper-lower; }

    anari::math::float3 center() const
    { return lower+size()*anari::math::float3(0.5f); }

    float volume() const
    { return size().x*size().y*size().z; }

    bool valid() const
    { return lower.x <= upper.x && lower.y <= upper.y && lower.z <= upper.z; }

    void invalidate()
    {
      lower.x = 1e31f;
      lower.y = 1e31f;
      lower.z = 1e31f;
      upper.x = -1e31f;
      upper.y = -1e31f;
      upper.z = -1e31f;
    }

    box3& extend(anari::math::float3 p) {
      lower.x = fminf(lower.x, p.x);
      lower.y = fminf(lower.y, p.y);
      lower.z = fminf(lower.z, p.z);

      upper.x = fmaxf(upper.x, p.x);
      upper.y = fmaxf(upper.y, p.y);
      upper.z = fmaxf(upper.z, p.z);

      return *this;
    }

    box3& extend(const box3 &b) {
      return extend(b.lower).extend(b.upper);
    }
  };

  struct box3i
  {
    box3i() = default;
    box3i(anari::math::int3 lo, anari::math::int3 up) : lower(lo), upper(up) {}

    anari::math::int3 lower, upper;

    anari::math::int3 size() const
    { return upper-lower; }

    anari::math::int3 center() const
    { return lower+size()/anari::math::int3(2); }

    long long volume() const
    { return size().x*(long long)size().y*size().z; }

    box3i& extend(anari::math::int3 p) {
      lower.x = min(lower.x, p.x);
      lower.y = min(lower.y, p.y);
      lower.z = min(lower.z, p.z);

      upper.x = max(upper.x, p.x);
      upper.y = max(upper.y, p.y);
      upper.z = max(upper.z, p.z);

      return *this;
    }

    box3i& extend(const box3i &b) {
      return extend(b.lower).extend(b.upper);
    }
  };
} // namespace anari::math
