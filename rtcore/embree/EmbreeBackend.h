#pragma once

#include "rtcore/common/Backend.h"

namespace barney {
  namespace embree {
    struct EmbreeBackend;
    
    struct DevGroup : public rtc::DevGroup {
      DevGroup(EmbreeBackend *backend,
               const std::vector<int> &gpuIDs,
               size_t sizeOfGlobals);
      rtc::Group *
      createTrianglesGroup(const std::vector<rtc::Geom *> &geoms)
        override
      { BARNEY_NYI(); };
      
      rtc::Group *
      createUserGeomsGroup(const std::vector<rtc::Geom *> &geoms) 
        override
      { BARNEY_NYI(); };

      void free(rtc::Group *)
        override 
      { BARNEY_NYI(); };
      
    };

    struct EmbreeBackend : public rtc::Backend {
      EmbreeBackend();
      
      // void setActiveGPU(int physicalID) override;
      // int  getActiveGPU() override;
      rtc::DevGroup *createDevGroup(const std::vector<int> &gpuIDs,
                                    size_t sizeOfGlobals) override;
    };
    
  }
}

  
  
