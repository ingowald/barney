#include "rtcore/embree/Compute.h"

namespace barney {
  namespace embree {
    
    Compute::Compute(Device *device,
                     const std::string &name)
      : rtc::Compute(device)
    { BARNEY_NYI(); }

    Trace::Trace(Device *device,
                 const std::string &name,
                 size_t sizeOfRG)
      : rtc::Trace(device)
    { BARNEY_NYI(); }
    

  }
}
