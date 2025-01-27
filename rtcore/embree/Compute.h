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

#include "rtcore/embree/Device.h"

namespace barney {
  namespace embree {

    struct ComputeInterface;
    struct TraceInterface;
    
    struct Compute : public rtc::Compute
    {
      typedef void (*ComputeFct)(ComputeInterface &,
                                 const void *dd);

      Compute(Device *device, const std::string &name);

      void launch(int numBlocks,
                  int blockSize,
                  const void *dd) override;
      
      void launch(vec2i numBlocks,
                  vec2i blockSize,
                  const void *dd) override;
      
      void launch(vec3i numBlocks,
                  vec3i blockSize,
                          const void *dd) override;

      std::string const name;
      ComputeFct computeFct = 0;
    };
    
    struct Trace : public rtc::Trace
    {
      typedef void (*TraceFct)(TraceInterface &);
      Trace(Device *device,
            const std::string &name);
      
      void launch(vec2i launchDims,
                  const void *dd);      
      void launch(int launchDims,
                  const void *dd);
      
      void sync() override
      { /* no-op */ }
      TraceFct traceFct = 0;
    };

  }
}
