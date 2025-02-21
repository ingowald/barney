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

#include <array>

#include "barney/volume/Volume.h"

namespace BARNEY_NS {

  struct Volume;
  struct VolumeAccel;
  struct MCGrid;
  struct ModelSlot;

  /*! abstracts any sort of scalar field (unstructured, amr,
    structured, rbfs....) _before_ any transfer function(s) get
    applied to it */
  struct ScalarField : public SlottedObject
  {
    typedef std::shared_ptr<ScalarField> SP;

    /*! Device-side data common to all ScalarFields that live on the device */
    struct DD {
      /*! world bounds, CLIPPED TO DOMAIN (if non-empty domain is present!) */
      box3f                worldBounds;
    };
    DD getDD(Device *device) const { return { worldBounds }; }
    
    ScalarField(Context *context,
                const DevGroup::SP &devices,
                const box3f &domain=box3f());

    static ScalarField::SP create(Context *context,
                                  const DevGroup::SP &devices,
                                  const std::string &type);

    virtual std::shared_ptr<VolumeAccel> createAccel(Volume *volume) = 0;

    virtual void buildMCs(MCGrid &macroCells);

    box3f     worldBounds;
    
    /*! a clipping box used to restrict whatever primitives the volume
        may be made up to down to a specific 3d box. e.g., if there's
        ghost cells, or if this is a spatial partitioning of a umesh,
        etc */
    const box3f domain;
  };
  
  /*! abstraction for a class that can sample a given scalar
    field. it's up to that class to create the right sampler for its
    data, and to do that only for the kind of traversers/accels that
    actually need to be able to sample.

    For the device side, all the actual sampling functionality will be
    in the DD's of the derived classes; the parent class doesnt' even
    have a DD, because all the device-side sampling code will (have to
    be) resolved through templates, in which case the classes using
    the sampler will know the actual type of that sampler (and it's
    DD)
  */
  struct ScalarFieldSampler {
    virtual void build() = 0;
    struct DD {
      /* derived classes ned to implement:
         
         inline __both__ float sample(vec3f P, bool dbg)
         
      */
    };
  };
  
}

