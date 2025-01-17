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

#include "barney/light/EnvMap.h"
#include "barney/ModelSlot.h"
#include "barney/Context.h"

namespace barney {
  namespace EnvMap_kernels {

    /*! computes an importance sampling weight for each pixel; gets
        called with one thread per pixel, in a 2d launch */
    __global__ void computeWeights_xy(float *allLines_cdf_x,
                                      cudaTextureObject_t texture,
                                      vec2i textureDims)
    {
      int ix = threadIdx.x + blockIdx.x * blockDim.x;
      int iy = threadIdx.y + blockIdx.y * blockDim.y;

      if (ix >= textureDims.x) return;
      if (iy >= textureDims.y) return;
      
      auto importance = [&](float4 v)->float { return max(max(v.x,v.y),v.z); };
      
      // float4 fromTex = tex2D<float4>(texture,
      //                                (ix+.5f)/(textureDims.x),
      //                                (iy+.5f)/(textureDims.y));
      float weight = 0.f;
      for (int iiy=0;iiy<=2;iiy++)
        for (int iix=0;iix<=2;iix++) {
          float4 fromTex = tex2D<float4>(texture,
                                         (ix+iix*.5f)/(textureDims.x),
                                         (iy+iiy*.5f)/(textureDims.y));
          weight = max(weight,importance(fromTex));
        }
      // float weight = importance(fromTex);

      allLines_cdf_x[ix+textureDims.x*iy] = weight;
    }

    /*! this kernel does one thread per line, then this one thread does
      entire line. not great, but we're not doing this per frame,
      anyway */
    __global__ void computeCDFs_doLine(float *cdf_y,
                                       float *allLines_cdf_x,
                                       vec2i textureDims)
    {
      int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
      int y = tid;
      if (y >= textureDims.y) return;
      float *thisLine_pdf = allLines_cdf_x + y * textureDims.x;
    
      float sum = 0.f;
      for (int ix=0;ix<textureDims.x;ix++) 
        sum += thisLine_pdf[ix];

      float rcp_sum = 1.f/sum;
      sum = 0.f;
      for (int ix=0;ix<textureDims.x;ix++) {
        sum += thisLine_pdf[ix];
        thisLine_pdf[ix] = sum * rcp_sum;
      }
      thisLine_pdf[textureDims.x-1] = 1.f;

      float rel_y = (y+.5f) / textureDims.y;

      const float theta = ONE_PI * rel_y;
      
      float relativeWeightOfLine = sum * sinf(theta);
      cdf_y[y] = relativeWeightOfLine;
    }

    /*! run by a single thread, to normalize the cdf_y */
    __global__ void normalize_cdf_y(float *cdf_y,
                                    const float *allLines_cdf_x,
                                    vec2i textureDims)
    {
      if (threadIdx.x != 0) return;

      float sum = 0.f;
      for (int i=0;i<textureDims.y;i++)
        sum += cdf_y[i];
      float rcp_sum = 1.f/sum;
    
      sum = 0.f;
      for (int i=0;i<textureDims.y;i++) {
        sum += cdf_y[i];
        cdf_y[i] = sum * rcp_sum;
      }
      cdf_y[textureDims.y-1] = 1.f;
    }
  }
                                             
                                             
  EnvMapLight::DD EnvMapLight::getDD(const Device::SP &device) const
  {
    DD dd;
    dd.dims = dims;
    if (texture) {
      dd.texture
        // = owlTextureGetObject(texture,device->owlID);
        = texture->getDD(device->rtc);
      dd.cdf_y
        // = (const float *)owlBufferGetPointer(cdf_y,device->owlID);
        = (const float *)cdf_y->getDD(device->rtc);
      dd.allCDFs_x
        // = (const float *)owlBufferGetPointer(allCDFs_x,device->owlID);
        = (const float *)allCDFs_x->getDD(device->rtc);
    } else {
      dd.texture = 0;
      dd.cdf_y = 0;
      dd.allCDFs_x = 0;
    }
      
    dd.toWorld = toWorld;
    dd.toLocal = toLocal;
    return dd;
  }

  EnvMapLight::EnvMapLight(Context *context, int slot)
    : Light(context,slot)
  {
    // std::cout << OWL_TERMINAL_YELLOW
    //           << "#bn: created env-map light"
    //           << OWL_TERMINAL_DEFAULT << std::endl;
    auto rtc = getRTC();
    cdf_y
      // = owlDeviceBufferCreate(getOWL(),OWL_FLOAT,1,nullptr);
      = rtc->createBuffer(sizeof(float));
    allCDFs_x
      // = owlDeviceBufferCreate(getOWL(),OWL_FLOAT,1,nullptr);
      = rtc->createBuffer(sizeof(float));
  }
  
  void EnvMapLight::commit()
  {
    toWorld.vz = normalize(params.up);
    toWorld.vy = normalize(cross(toWorld.vz,params.direction));
    toWorld.vx = normalize(cross(toWorld.vy,toWorld.vz));
    toLocal    = rcp(toWorld);
    assert(params.texture);
    // texture    = params.texture->owlTexture;
    texture    = params.texture->rtcTexture;
    computeCDFs();
  }


  void EnvMapLight::computeCDFs()
  {
    // std::cout << "#bn: computing env-map CDFs" << std::endl;
    assert(texture);
    dims = (const vec2i &)texture->getDims();//owlTextureGetDimensions(texture);
    auto rtc = getRTC();
    // owlBufferResize(cdf_y,dims.y);
    rtc->free(cdf_y);
    cdf_y = rtc->createBuffer(dims.y*sizeof(float));
    // owlBufferResize(allCDFs_x,dims.x*dims.y);
    rtc->free(allCDFs_x);
    allCDFs_x = rtc->createBuffer(dims.x*dims.y*sizeof(float));
    
    for (auto device : getDevices()) {
      SetActiveGPU forThisKernel(device);

#if 1
      BARNEY_NYI();
#else
      BARNEY_CUDA_SYNC_CHECK();
      
      /* computes an importance sampling weight for each pixel; gets
         called with one thread per pixel, in a 2d launch */
      vec2i bs = 16;
      vec2i nb = divRoundUp(dims,bs);
      CHECK_CUDA_LAUNCH(EnvMap_kernels::computeWeights_xy,
                        nb,bs,0,0,
                        //
                        (float*)owlBufferGetPointer(allCDFs_x,device->owlID),
                        owlTextureGetObject(texture,device->owlID),
                        dims);
      // EnvMap_kernels::computeWeights_xy<<<nb,bs>>>
      //   ((float*)owlBufferGetPointer(allCDFs_x,device->owlID),
      //    owlTextureGetObject(texture,device->owlID),
      //    dims);
      BARNEY_CUDA_SYNC_CHECK();

      /* this kernel will do one thread per line, then this one thread does
         entire line. not great, but we're not doing this per frame,
         anyway */
      CHECK_CUDA_LAUNCH(EnvMap_kernels::computeCDFs_doLine,
                        dims.y,1024,0,0,
                        (float*)owlBufferGetPointer(cdf_y,device->owlID),
                        (float*)owlBufferGetPointer(allCDFs_x,device->owlID),
                        dims);
      // EnvMap_kernels::computeCDFs_doLine<<<dims.y,1024>>>
      //   ((float*)owlBufferGetPointer(cdf_y,device->owlID),
      //    (float*)owlBufferGetPointer(allCDFs_x,device->owlID),
      //    dims);
      BARNEY_CUDA_SYNC_CHECK();
      
      /* run by a single thread, to normalize the cdf_y */
      CHECK_CUDA_LAUNCH(EnvMap_kernels::normalize_cdf_y,
                        1,1,0,0,
                        //
                        (float*)owlBufferGetPointer(cdf_y,device->owlID),
                        (float*)owlBufferGetPointer(allCDFs_x,device->owlID),
                        dims);
      // EnvMap_kernels::normalize_cdf_y<<<1,1>>>
      //   ((float*)owlBufferGetPointer(cdf_y,device->owlID),
      //    (float*)owlBufferGetPointer(allCDFs_x,device->owlID),
      //    dims);

      BARNEY_CUDA_SYNC_CHECK();
#endif
    }
    // std::cout << "#bn: computing env-map CDFs .... done." << std::endl;
  }
  
  
  bool EnvMapLight::set2i(const std::string &member, const vec2i &value) 
  {
    return false;
  }

  bool EnvMapLight::set3f(const std::string &member, const vec3f &value) 
  {
    if (member == "direction") {
      params.direction = value;
      return true;
    }
    if (member == "up") {
      params.up = value;
      return true;
    }
    return false;
  }

  bool EnvMapLight::setObject(const std::string &member, const Object::SP &value) 
  {
    if (member == "texture") {
      params.texture = value->as<Texture>();
      // params.texture = texture ? texture->owlTex : 0;
      return true;
    }
    return false;
  }

}
