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

#undef pi

#include <nanovdb/GridHandle.h>
#include <nanovdb/cuda/DeviceBuffer.h>
#include <nanovdb/math/SampleFromVoxels.h>

#include "barney/ModelSlot.h"

namespace barney {

  struct NanoVDBField : public ScalarField
  {
    typedef std::shared_ptr<NanoVDBField> SP;

    struct DD : public ScalarField::DD {

      static void addVars(std::vector<OWLVarDecl> &vars, int base);

      nanovdb::NanoGrid<float> *gridPtr;
      enum FilterMode { Nearest, Linear };
      FilterMode filterMode;
    };

    void setVariables(OWLGeom geom) override;

    void buildMCs(MCGrid &macroCells) override;

    NanoVDBField(Context *context, int slot,
                 std::vector<float> &gridData);

    DD getDD(const Device::SP &device);

    VolumeAccel::SP createAccel(Volume *volume) override;

    typedef nanovdb::GridHandle<nanovdb::cuda::DeviceBuffer> GridHandle;
    // one nvdb grid handle per-device:
    std::vector<GridHandle> gridHandles;
  };

  /*! sampler type for nanovdb; this is essentially just a wrapper
      for a nvdb grid such as the field already has, but with a sample
      interface */
  struct NanoVDBSampler {
    struct DD : public NanoVDBField::DD {
      inline __device__ float sample(const vec3f P, bool dbg) const
      {
        //if (dbg) printf("%f,%f,%f\n",P.x,P.y,P.z);
        auto acc = gridPtr->getAccessor();
        auto PP = P;
        if (1) {
          auto smp = nanovdb::math::createSampler<1>(acc);
          float f = smp(nanovdb::math::Vec3<float>(PP.x,PP.y,PP.z));
          return f;
        }
        else {
          auto smp = nanovdb::math::createSampler<0>(acc);
          float f = smp(nanovdb::math::Vec3<float>(PP.x,PP.y,PP.z));
          return f;
        }
      }
    };

    struct Host {
      Host(ScalarField *sf) : field((NanoVDBField *)sf) {}

      /*! builds the string that allows for properly matching optix
          device progs for this type */
      inline std::string getTypeString() const { return "NanoVDB"; }

      void build(bool full_rebuild) { /* nothing to do for this class */}

      void setVariables(OWLGeom geom) { /* nothing to do for this class */}

      NanoVDBField *const field;
    };
  };
}
