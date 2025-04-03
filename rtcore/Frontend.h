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

#if BARNEY_RTC_OPTIX
# include "rtcore/optix/Device.h"
# include "rtcore/optix/Geom.h" 
# include "rtcore/optix/Group.h" 
# include "rtcore/optix/Denoiser.h" 
# include "rtcore/cudaCommon/Texture.h"
# include "rtcore/cudaCommon/TextureData.h"
# include "rtcore/cudaCommon/ComputeKernel.h"
// # include "rtcore/optix/TraceInterface.h" 

namespace rtc {
  namespace optix {
    /*! forward declaration to allow defining functions with it - only
        device programs should ever include the 'real'
        rtcore/TraceInterface.h */
    struct TraceInterface;
  }
  using namespace optix;
  
  using rtc::cuda_common::ComputeKernel1D;
  using rtc::cuda_common::ComputeKernel2D;
  using rtc::cuda_common::ComputeKernel3D;

  using rtc::cuda_common::float2;
  using rtc::cuda_common::float3;
  using rtc::cuda_common::float4;
  using rtc::cuda_common::int2;
  using rtc::cuda_common::int3;
  using rtc::cuda_common::int4;
  
  using rtc::cuda_common::load;

  using rtc::optix::TraceInterface;
  using ::rtc::device::TextureObject;
  
# ifdef __CUDACC__
#  define RTC_DEVICE_CODE 1
# endif
}
#endif



#if BARNEY_RTC_EMBREE
# include "rtcore/common/rtcore-common.h"
# include "rtcore/embree/Device.h"
# include "rtcore/embree/Geom.h" 
# include "rtcore/embree/GeomType.h" 
# include "rtcore/embree/Group.h" 
# include "rtcore/embree/Denoiser.h" 
# include "rtcore/embree/ComputeInterface.h" 
# include "rtcore/embree/ComputeKernel.h"

namespace rtc {
  namespace embree {
    /*! forward declaration to allow defining functions with it - only
        device programs should ever include the 'real'
        rtcore/TraceInterface.h */
    struct TraceInterface;
  }
  using namespace rtc::embree;

  using rtc::embree::ComputeInterface;
  using rtc::embree::ComputeKernel1D;
  using rtc::embree::ComputeKernel2D;
  using rtc::embree::ComputeKernel3D;
  using rtc::embree::load;
  using rtc::embree::tex1D;
  using rtc::embree::tex2D;
  using rtc::embree::tex3D;
  using rtc::embree::TraceInterface;
  using ::rtc::device::TextureObject;

# define RTC_DEVICE_CODE 1
}
#endif


#if BARNEY_RTC_CUDA
# include "rtcore/cuda/Device.h"
# include "rtcore/cuda/Geom.h" 
# include "rtcore/cuda/GeomType.h" 
# include "rtcore/cuda/Group.h" 
# include "rtcore/cuda/TraceKernel.h" 
# include "rtcore/cudaCommon/Texture.h"
# include "rtcore/cudaCommon/TextureData.h"
# include "rtcore/cudaCommon/ComputeKernel.h"
# include "rtcore/cudaCommon/ComputeInterface.h"

namespace rtc {
  namespace cuda {
    /*! forward declaration to allow defining functions with it - only
        device programs should ever include the 'real'
        rtcore/TraceInterface.h */
    struct TraceInterface;
  }
  using namespace rtc::cuda;

  using rtc::cuda_common::ComputeKernel1D;
  using rtc::cuda_common::ComputeKernel2D;
  using rtc::cuda_common::ComputeKernel3D;

  using rtc::cuda_common::float2;
  using rtc::cuda_common::float3;
  using rtc::cuda_common::float4;
  using rtc::cuda_common::int2;
  using rtc::cuda_common::int3;
  using rtc::cuda_common::int4;
  
  using rtc::cuda_common::load;

  using rtc::cuda::TraceInterface;

#if RTC_DEVICE_CODE
  using rtc::cuda_common::tex1D;
  using rtc::cuda_common::tex2D;
  using rtc::cuda_common::tex3D;
  using rtc::cuda_common::fatomicMin;
  using rtc::cuda_common::fatomicMax;
#endif
  
}
#endif



namespace rtc {
  using device::TextureObject;
}
