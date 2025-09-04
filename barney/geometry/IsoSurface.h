// ======================================================================== //
// Copyright 2023-2024 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#pragma once

#include "barney/geometry/Geometry.h"
#include "barney/volume/Volume.h"

namespace BARNEY_NS {

  struct IsoSurface;
  
  struct IsoSurfaceAccel {
    typedef std::shared_ptr<IsoSurfaceAccel> SP;

    IsoSurfaceAccel(IsoSurface *isoSurface);
    
    virtual void build(bool full_rebuild) = 0;
  };
  
  struct IsoSurface : public Geometry {
    typedef std::shared_ptr<IsoSurface> SP;

    template<typename SFSampler>
    struct DD {
      inline __rtc_device
      vec4f sample(vec3f point, bool dbg=false) const
      {
        return sfSampler.sample(point,dbg);
      }
      
      ScalarField::DD        sfCommon;
      typename SFSampler::DD sfSampler;
      TransferFunction::DD   xf;
      int                    userID;
    };

    IsoSurface(Context *context, DevGroup::SP devices);
    
    /*! pretty-printer for printf-debugging */
    std::string toString() const override
    { return "IsoSurface{}"; }

    void commit() override;
    
    // ------------------------------------------------------------------
    /*! @{ parameter set/commit interface */
    bool set1f(const std::string &member, const float &value) override;
    bool setData(const std::string &member, const barney_api::Data::SP &value) override;
    bool setObject(const std::string &member, const Object::SP &value) override;
    /*! @} */
    // ------------------------------------------------------------------
    
    float            isoValue  = NAN;
    PODData::SP      isoValues = 0;
    ScalarField::SP  sf;
    VolumeAccel::SP  accel;
  };
  
}
  
