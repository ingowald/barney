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

    using cuda_common::Texture;
    using cuda_common::TextureData;
    using cuda_common::SetActiveGPU;

    struct Buffer;
    struct Device;
    struct Geom;
    struct GeomType;
    
    struct TraceKernel2D {
      TraceKernel2D(Device *device,
                    const std::string &ptxCode,
                    const std::string &kernelName,
                    size_t sizeOfLP);
      void launch(vec2i launchDims,
                  const void *kernelData);
      Device *const device;
    };
    
    struct Device : public cuda_common::Device{
      Device(int physicalGPU)
        : cuda_common::Device(physicalGPU)
      {}
      
    };

  }
}
