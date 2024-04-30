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

#include "barney/Object.h"
#include "barney/common/mat4.h"
#include "barney/render/HitAttributes.h"
// #include "barney/material/device/Material.h"
#include "barney/render/Sampler.h"

namespace barney {
  namespace render {

    struct DeviceMaterial;
    
    struct PossiblyMappedParameter {
      typedef enum { INVALID=0, VALUE, ATTRIBUTE, SAMPLER// , ARRAY
      } Type;
      
      struct DD {
        inline __device__
        float4 eval(const HitAttributes &hitData, bool dbg=false) const;
        
        Type type;
        union {
          float4               value;
          HitAttributes::Which attribute;
          int                  samplerID;
          // struct {
          //   BNDataType  elementType;
          //   const void *pointer;
          // } array;
        };
      };

      void set(const vec3f  &v);
      void set(const float4 &v);
      void set(Sampler::SP sampler);
      // void set(PODData::SP array);
      void set(const std::string &attributeName);
      void make(DD &dd, int deviceID) const;
      
      Type type = VALUE;
      Sampler::SP          sampler;
      // PODData::SP          array;
      HitAttributes::Which attribute;
      float4               value { 0.f, 0.f, 0.f, 1.f };
    };

    /*! barney 'virtual' material implementation that takes anari-like
      material paramters, and then builder barney::render:: style
      device materials to be put into the device geometries */
    struct HostMaterial : public SlottedObject {
      typedef std::shared_ptr<HostMaterial> SP;

      /*! pretty-printer for printf-debugging */
      std::string toString() const override { return "<Material>"; }

      /*! device-data, as a union of _all_ possible device-side
        materials; we have to use a union here because no matter what
        virtual barney::Material gets created on the host, we have to
        have a single struct we put into the OWLGeom/SBT entry, else
        we'd have to have different OWLGeom type for different
        materials .... and possibly even change the actual OWLGeom
        (and even worse, its type) if the assigned material's type
        changes */
      // using DD = barney::render::DeviceMaterial;

      HostMaterial(ModelSlot *owner);
      virtual ~HostMaterial();

      // ------------------------------------------------------------------
      /*! @{ parameter set/commit interface */
      void commit() override;
      /*! @} */
      // ------------------------------------------------------------------
      static HostMaterial::SP create(ModelSlot *dg, const std::string &type);
    
      void setDeviceDataOn(OWLGeom geom) const;
    
      virtual void createDD(DeviceMaterial &dd, int deviceID) const = 0;

      /*! declares the device-data's variables to an owl geom */
      static void addVars(std::vector<OWLVarDecl> &vars, int base);

      const int materialID;
    };

    inline __device__
    float4 PossiblyMappedParameter::DD::eval(const HitAttributes &hitData, bool dbg) const
    {
      if (dbg)
        printf("evaluating attrib, type %i\n",int(type));
      if (type == VALUE)
        return value;
      if (type == ATTRIBUTE) {
        if (dbg) printf("asking hitattributes for attribute %i\n",(int)attribute);
        return hitData.get(attribute,dbg);
      }
      printf("un-handled material input type %i\n",(int)type);
      return make_float4(0.f,0.f,0.f,1.f);
    }

  }
}
