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

namespace barney {
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
      // if (colorBuf)  oidnReleaseBuffer(colorBuf);
      // if (normalBuf) oidnReleaseBuffer(normalBuf);
      // if (outputBuf) oidnReleaseBuffer(outputBuf);
      // oidnReleaseFilter(filter);
      // oidnReleaseDevice(device);
    }
    
    void Denoiser::resize(vec2i size)
    {
      this->numPixels = size;
      // if (colorBuf)  oidnReleaseBuffer(colorBuf);
      // if (normalBuf) oidnReleaseBuffer(normalBuf);
      // if (outputBuf) oidnReleaseBuffer(outputBuf);
      // colorBuf
      //   = oidnNewSharedBuffer(device, fb->linearColor,
      //                         numPixels.x*numPixels.y*sizeof(vec4f));
      // normalBuf
      //   = oidnNewSharedBuffer(device, fb->linearNormal,
      //                         numPixels.x*numPixels.y*sizeof(vec3f));
      // oidnSetFilterImage(filter,"color",colorBuf,
      //                    OIDN_FORMAT_FLOAT4,numPixels.x,numPixels.y,0,0,0);
      // oidnSetFilterImage(filter,"output",outputBuf,
      //                    OIDN_FORMAT_FLOAT3,numPixels.x,numPixels.y,0,0,0);
      // oidnSetFilterBool(filter,"hdr",true);
      // oidnCommitFilter(filter);
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
      // oidnSetSharedFilterImage(filter,"color",in_rgba,
      //                          OIDN_FORMAT_FLOAT4,numPixels.x,numPixels.y,0,0,0);
      // oidnSetSharedFilterImage(filter,"normal",in_normal,
      //                          OIDN_FORMAT_FLOAT3,numPixels.x,numPixels.y,0,0,0);
      
      oidnSetSharedFilterImage(filter,"output",out_rgba,
                               OIDN_FORMAT_FLOAT3,numPixels.x,numPixels.y,0,
                               sizeof(vec4f),0);
      // oidnSetSharedFilterImage(filter,"output",out_rgba,
      //                          OIDN_FORMAT_FLOAT4,numPixels.x,numPixels.y,0,0,0);
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
