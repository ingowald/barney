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

#include "rtcore/embree/Denoiser.h"

#if BARNEY_OIDN_CPU

namespace rtc {
  namespace embree {

    Denoiser::Denoiser(Device *device)
      : rtc(device)
    {
      oidnDevice = 
        oidnNewDevice(OIDN_DEVICE_TYPE_CPU);
      oidnCommitDevice(oidnDevice);
      
      filter = oidnNewFilter(oidnDevice,"RT");
    }

    Denoiser::~Denoiser()
    {
      oidnReleaseFilter(filter);
      oidnReleaseDevice(oidnDevice);
    }
    
    void Denoiser::resize(vec2i size)
    {
      this->numPixels = size;
    }
    
    void Denoiser::run(// output
                       vec4f *out_rgba,
                       // input channels
                       vec4f *in_rgba,
                       vec3f *in_normal,
                       float blendFactor)
    {
      oidnSetSharedFilterImage(filter,"color",in_rgba,
                               OIDN_FORMAT_FLOAT3,numPixels.x,numPixels.y,0,
                               sizeof(vec4f),0);
#if 1
      // no normal channel - oidn complains about 'unsupported
      // combination' if we pass this.
#else
      oidnSetSharedFilterImage(filter,"normal",in_normal,
                                OIDN_FORMAT_FLOAT3,numPixels.x,numPixels.y,0,
                                sizeof(vec3f),0);
#endif
      oidnSetSharedFilterImage(filter,"output",out_rgba,
                               OIDN_FORMAT_FLOAT3,numPixels.x,numPixels.y,0,
                               sizeof(vec4f),0);
      oidnSetFilterBool(filter,"hdr",true);
      oidnCommitFilter(filter);

      
      // copy input to output, to copy the alpha channel (oidn doesn't do alpha)
      memcpy(out_rgba,in_rgba,numPixels.x*numPixels.y*sizeof(vec4f));
      // and run the filter on (strided) rgba
      oidnExecuteFilter(filter);
      const char *error;
      oidnGetDeviceError(oidnDevice,&error);
      if (error)
        PRINT(error);
    }
    
  }
}
#endif
