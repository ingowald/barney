#include "rtcore/embree/Compute.h"

namespace barney {
  namespace embree {
    
    Compute::Compute(Device *device,
                     const std::string &name)
      : rtc::Compute(device)
    {
      computeFct = (ComputeFct)rtc::getSymbol
        ("barney_rtc_embree_compute_"+name);
    }

    Trace::Trace(Device *device,
                 const std::string &name,
                 size_t sizeOfRG)
      : rtc::Trace(device)
    { BARNEY_NYI(); }
    

  }
}
