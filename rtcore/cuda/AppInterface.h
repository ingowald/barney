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

#include "rtcore/cuda/Device.h"
#include "rtcore/cuda/Buffer.h"
#include "rtcore/cudaCommon/ComputeKernel.h"
#include "rtcore/cuda/TraceKernel.h"
// for setPrimCount etc
#include "rtcore/cuda/Geom.h"
// for buildAccel
#include "rtcore/cuda/Group.h"
// for createTexture
#include "rtcore/cudaCommon/TextureData.h"
// for getDD
#include "rtcore/cudaCommon/Texture.h"

namespace rtc {
  namespace cuda {

    using rtc::cuda_common::enablePeerAccess;
    
    using rtc::cuda_common::ComputeKernel1D;
    using rtc::cuda_common::ComputeKernel2D;
    using rtc::cuda_common::ComputeKernel3D;
    
    using rtc::cuda_common::Texture;
    using rtc::cuda_common::TextureData;
  }
}

#define RTC_IMPORT_USER_GEOM(moduleName,typeName,DD,has_ah,has_ch)      \
  extern ::rtc::cuda::GeomType *createGeomType_##typeName(::rtc::Device *);

#define RTC_IMPORT_TRIANGLES_GEOM(moduleName,typeName,DD,has_ah,has_ch) \
  extern rtc::cuda::GeomType *createGeomType_##typeName(rtc::Device *);






  
