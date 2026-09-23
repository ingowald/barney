#pragma once

namespace BARNEY_NS {
  namespace native {

    inline __rtc_both float remapImageRegionCoordinate(float lower,
                                                       float upper,
                                                       float screen)
    {
      return lower + screen * (upper - lower);
    }

  }
}
