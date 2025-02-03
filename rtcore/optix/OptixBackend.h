// ======================================================================== //
// Copyright 2023-2025 Ingo Wald                                            //
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

#include "rtcore/optix/Device.h"

namespace barney {
  namespace optix {
    struct OptixBackend;

    struct Buffer : public rtc::Buffer {
      Buffer(optix::Device *device,
             size_t size,
             const void *initData);
      void *getDD() const override;
      
      OWLBuffer owl;
    };

    struct Group : public rtc::Group {
      Group(optix::Device *device, OWLGroup owlGroup);
      virtual ~Group() { owlGroupRelease(owl); }
      
      rtc::device::AccelHandle getDD() const override;
      void buildAccel() override;
      void refitAccel() override;
      
      OWLGroup const owl;
    };

    struct GeomType;
    
    struct Geom : public rtc::Geom {
      Geom(GeomType *gt, OWLGeom geom);
      virtual ~Geom();
      void setDD(const void *dd) override;
      
      OWLGeom const owl;
    };

    struct TrianglesGeom : public Geom {
      TrianglesGeom(GeomType *gt, OWLGeom geom);
      
      /*! only for user geoms */
      void setPrimCount(int primCount) override;
      /*! can only get called on triangle type geoms */
      void setVertices(rtc::Buffer *vertices, int numVertices) override;
      void setIndices(rtc::Buffer *indices, int numIndices) override;
    };
    struct UserGeom : public Geom {
      UserGeom(GeomType *gt, OWLGeom geom);
      
      /*! only for user geoms */
      void setPrimCount(int primCount) override;
      /*! can only get called on triangle type geoms */
      void setVertices(rtc::Buffer *vertices, int numVertices) override;
      void setIndices(rtc::Buffer *indices, int numIndices) override;
    };
    
    struct GeomType : public rtc::GeomType {
      GeomType(optix::Device *device) : rtc::GeomType(device) {}
      virtual ~GeomType() override;
      
      OWLGeomType gt = 0;
    };
    struct TrianglesGeomType : public GeomType
    {
      TrianglesGeomType(optix::Device *device,
                        const std::string &ptxName,
                        const std::string &typeName,
                        size_t sizeOfDD, bool has_ah, bool has_ch);
      rtc::Geom *createGeom() override;
    };
    struct UserGeomType : public GeomType
    {
      UserGeomType(optix::Device *device,
                   const std::string &ptxName,
                   const std::string &typeName,
                   size_t sizeOfDD, bool has_ah, bool has_ch);
      rtc::Geom *createGeom() override;
    };
    
    struct OptixBackend : public cuda::BaseBackend {
      // setActive and getActive: inherited from cuda bacekdn
      OptixBackend();
      virtual ~OptixBackend() = default;
      
      rtc::Device *createDevice(int gpuID) override;
    };
    
  }
}

  
  
