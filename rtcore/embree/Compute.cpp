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

    void Compute::launch(int numBlocks,
                         int blockSize,
                         const void *dd)
    { BARNEY_NYI(); }

    void Compute::launch(vec2i numBlocks,
                         vec2i blockSize,
                         const void *dd)
    { BARNEY_NYI(); }

    void Compute::launch(vec3i numBlocks,
                         vec3i blockSize,
                         const void *dd)
    { BARNEY_NYI(); }
      

    Trace::Trace(Device *device,
                 const std::string &name,
                 size_t sizeOfRG)
      : rtc::Trace(device)
    { 
      traceFct = (TraceFct)rtc::getSymbol
        ("barney_rtc_embree_trace_"+name);
    }
    
    void Trace::launch(vec2i launchDims,
                       const void *dd) 
    { BARNEY_NYI(); }
      
    void Trace::launch(int launchDims,
                       const void *dd) 
    { BARNEY_NYI(); }

  }
}
