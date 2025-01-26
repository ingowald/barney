#include "rtcore/embree/Compute.h"
#include "rtcore/common/RTCore.h"

namespace barney {
  namespace embree {
    
    Compute::Compute(Device *device,
                     const std::string &name)
      : rtc::Compute(device),
        name(name)
    {
      computeFct = (ComputeFct)rtc::getSymbol
        ("barney_rtc_embree_computeBlock_"+name);
    }

    void Compute::launch(int numBlocks,
                         int blockSize,
                         const void *dd)
    {
      launch(vec3i(numBlocks,1,1),
             vec3i(blockSize,1,1),
             dd);
    }

    void Compute::launch(vec2i numBlocks,
                         vec2i blockSize,
                         const void *dd)
    {
      launch(vec3i(numBlocks.x,numBlocks.y,1),
             vec3i(blockSize.x,blockSize.y,1),
             dd);
    }

    void Compute::launch(vec3i numBlocks,
                         vec3i blockSize,
                         const void *dd)
    {
      for (int bz=0;bz<numBlocks.z;bz++)
        for (int by=0;by<numBlocks.y;by++)
          for (int bx=0;bx<numBlocks.x;bx++)
            {
              embree::ComputeInterface ci;
              ci.gridDim = vec3ui(numBlocks);//vec3ui(numBlocks,1,1);
              ci.blockIdx = vec3ui(bx,by,bz);//vec3ui(b,0,0);
              ci.blockDim = vec3ui(blockSize);//vec3ui(blockSize,1,1);
              ci.threadIdx = vec3ui(0);
              computeFct(ci,dd);
            }
    }
      

    Trace::Trace(Device *device,
                 const std::string &name)
      : rtc::Trace(device)
    { 
      traceFct = (TraceFct)rtc::getSymbol
        ("barney_rtc_embree_trace_"+name);
    }
    
    void Trace::launch(vec2i launchDims,
                       const void *dd) 
    {
      for (int iy=0;iy<launchDims.y;iy++)
        for (int ix=0;ix<launchDims.x;ix++)
          {
            embree::TraceInterface ci;
            ci.launchIndex = vec3i(ix,iy,0);
            ci.launchDimensions = {launchDims.x,launchDims.y,1};
            ci.lpData = dd;
            traceFct(ci);
          }
    }
      
    void Trace::launch(int launchDims,
                       const void *dd) 
    {
      launch(vec2i(launchDims,1),dd);
    }

  }
}
