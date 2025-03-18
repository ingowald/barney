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

#include "rtcore/cudaCommon/Device.h"

namespace rtc {
  namespace cuda {
    struct Geom;
    struct TraceInterface;
    
    typedef void (*BoundsProg)(const TraceInterface &,
                               const void *dd,
                               box3f &bounds,
                               int primID);
    typedef void (*IntersectProg)(TraceInterface &ti);
    typedef void (*AHProg)(TraceInterface &ti);
    typedef void (*CHProg)(TraceInterface &ti);
    
    struct GeomType
    {
      GeomType(Device *device, size_t sizeOfDD);
               
      virtual Geom *createGeom() = 0;
               
      Device *const device;
      size_t  const sizeOfDD;
    };

    struct UserGeomType : public GeomType {
      UserGeomType(Device *device,
                   size_t sizeOfDD,
                   BoundsProg bounds,
                   IntersectProg intersect,
                   AHProg ah,
                   CHProg ch);
      Geom *createGeom() override;

      BoundsProg const bounds;
      IntersectProg const intersect;
      AHProg const ah;
      CHProg const ch;
    };

    struct TrianglesGeomType : public GeomType {
      TrianglesGeomType(Device *device,
                        size_t sizeOfDD,
                        AHProg ah,
                        CHProg ch);
      Geom *createGeom() override;

      AHProg const ah;
      CHProg const ch;
    };
  }
}

#define RTC_IMPORT_USER_GEOM(moduleName,typeName,DD,has_ah,has_ch)      \
  extern ::rtc::GeomType *createGeomType_##typeName(::rtc::Device *);

#define RTC_IMPORT_TRIANGLES_GEOM(moduleName,typeName,DD,has_ah,has_ch) \
  extern rtc::GeomType *createGeomType_##typeName(rtc::Device *);



#define RTC_EXPORT_USER_GEOM(name,DD,Programs,has_ah,has_ch)    \
  ::rtc::GeomType *createGeomType_##name(::rtc::Device *device) \
  {                                                             \
    return new ::rtc::cuda::UserGeomType                        \
      (device,                                                  \
       sizeof(DD),                                              \
       Programs::bounds,                                        \
       Programs::intersect,                                     \
       has_ah?Programs::anyHit:0,                               \
       has_ch?Programs::closestHit:0);                          \
  }


#define RTC_EXPORT_TRIANGLES_GEOM(name,DD,Programs,has_ah,has_ch)       \
  __global__                                                            \
  void rtc_cuda_writeAddresses_##name(rtc::Geom::SBTHeader *h)          \
  {                                                                     \
    h->ah = has_ah?Programs::anyHit:0;                                  \
    h->ch = has_ch?Programs::closestHit:0;                              \
  }                                                                     \
  rtc::GeomType *createGeomType_##name(rtc::Device *device)             \
  {                                                                     \
    ::rtc::SetActiveGPU forDuration(device);                            \
    rtc::Geom::SBTHeader *h;                                            \
    cudaMalloc((void **)&h,sizeof(*h));                                 \
    rtc_cuda_writeAddresses_##name<<<1,32>>>(h);                        \
    device->sync();                                                     \
    rtc::Geom::SBTHeader hh;                                            \
    cudaMemcpy(&hh,h,sizeof(hh),cudaMemcpyDefault);                     \
    return new rtc::cuda::TrianglesGeomType                             \
      (device,                                                          \
       sizeof(DD),                                                      \
       hh.ah,                                                           \
       hh.ch);                                                          \
  }

