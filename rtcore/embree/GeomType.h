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

#include "rtcore/common/Backend.h"

namespace barney {
  namespace embree {
    struct TraceInterface;
    struct Device;
    
    typedef void (*AnyHitFct)(embree::TraceInterface &ti);
    typedef void (*ClosestHitFct)(embree::TraceInterface &ti);
    typedef void (*IntersectFct)(embree::TraceInterface &ti);
    typedef void (*BoundsFct)(embree::TraceInterface &ti,
                              const void *gt, box3f &result, int primID);


    struct GeomType : public rtc::GeomType
    {
      GeomType(Device *device,
               const std::string &name,
               size_t sizeOfProgramData,
               bool has_ah,
               bool has_ch);
      AnyHitFct ah = 0;
      ClosestHitFct ch = 0;
      size_t const sizeOfProgramData;
    };
      
    struct TrianglesGeomType : public GeomType
    {
      TrianglesGeomType(Device *device,
                        const std::string &name,
                        size_t sizeOfProgramData,
                        bool has_ah,
                        bool has_ch);
      
      virtual ~TrianglesGeomType();
      
      rtc::Geom *createGeom() override;
    };

    struct UserGeomType : public GeomType
    {
      UserGeomType(Device *device,
                   const std::string &name,
                   size_t sizeOfProgramData,
                   bool has_ah,
                   bool has_ch);
      virtual ~UserGeomType();
      
      rtc::Geom *createGeom() override;
      
      BoundsFct     bounds = 0;
      IntersectFct  intersect = 0;
    };
    
  }
}
