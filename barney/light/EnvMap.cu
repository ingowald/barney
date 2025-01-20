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

  /*! computes an importance sampling weight for each pixel; gets
    called with one thread per pixel, in a 2d launch */
  struct ComputeWeights_xy {
    float *allLines_cdf_x;
    rtc::device::TextureObject texture;
    vec2i textureDims;
    
    template<typename RTComputeInterface>
    inline __both__
    void run(const RTComputeInterface &ci)
    {
      int ix = ci.getThreadIdx().x
        + ci.getBlockIdx().x
        * ci.getBlockDim().x;
      int iy = ci.getThreadIdx().y
        + ci.getBlockIdx().y
        * ci.getBlockDim().y;
        
      if (ix >= textureDims.x) return;
      if (iy >= textureDims.y) return;
        
      auto importance = [&](float4 v)->float
      { return max(max(v.x,v.y),v.z); };
        
      float weight = 0.f;
      for (int iiy=0;iiy<=2;iiy++)
        for (int iix=0;iix<=2;iix++) {
          float4 fromTex
            = rtc::tex2D<float4>(texture,
                                 (ix+iix*.5f)/(textureDims.x),
                                 (iy+iiy*.5f)/(textureDims.y));
          weight = max(weight,importance(fromTex));
        }
        
      allLines_cdf_x[ix+textureDims.x*iy] = weight;
    }
  };

  /*! this kernel does one thread per line, then this one thread does
    entire line. not great, but we're not doing this per frame,
    anyway */
  struct ComputeCDFs_doLine {
    float *cdf_y;
    float *allLines_cdf_x;
    vec2i textureDims;

    template<typename RTComputeInterface>
    inline __both__
    void run(const RTComputeInterface &ci)
    {
      int tid
        = ci.getThreadIdx().x
        + ci.getBlockIdx().x * ci.getBlockDim().x;
        
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
  };

  /*! run by a single thread, to normalize the cdf_y */
  struct Normalize_cdf_y {
    float       *cdf_y;
    const float *allLines_cdf_x;
    vec2i        textureDims;
    
    template<typename RTComputeInterface>
    inline __both__
    void run(const RTComputeInterface &ci)
    {
      if (ci.getThreadIdx().x != 0) return;
        
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
  };
    
  EnvMapLight::DD EnvMapLight::getDD(const Device::SP &device) const
  {
    DD dd;
    dd.dims = dims;
    if (texture) {
      dd.texture
        = texture->getDD(device->rtc);
      dd.cdf_y
        = (const float *)cdf_y->getDD(device->rtc);
      dd.allCDFs_x
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
    dims = (const vec2i &)texture->getDims();
    auto rtc = getRTC();
    rtc->free(cdf_y);
    cdf_y = rtc->createBuffer(dims.y*sizeof(float));
    rtc->free(allCDFs_x);
    allCDFs_x = rtc->createBuffer(dims.x*dims.y*sizeof(float));
    
    for (auto device : getDevices()) {
      SetActiveGPU forThisKernel(device);
      
      /* computes an importance sampling weight for each pixel; gets
         called with one thread per pixel, in a 2d launch */
      {
        ComputeWeights_xy kernelData = {
          (float*)allCDFs_x->getDD(device->rtc),
          texture->getDD(device->rtc),
          dims
        };
        vec2i bs = 16;
        vec2i nb = divRoundUp(dims,bs);
        computeWeights_xy->launch(device->rtc,nb,bs,&kernelData);
      }
      /* this kernel will do one thread per line, then this one thread does
         entire line. not great, but we're not doing this per frame,
         anyway */
      {
        ComputeCDFs_doLine kernelData = {
          (float*)cdf_y->getDD(device->rtc),
          (float*)allCDFs_x->getDD(device->rtc),
          dims
        };
        computeCDFs_doLine->launch(device->rtc,
                                   dims.y,1024,
                                   &kernelData);
      }
      /* run by a single thread, to normalize the cdf_y */
      {
        Normalize_cdf_y kernelArgs = {
          (float*)cdf_y->getDD(device->rtc),
          (float*)allCDFs_x->getDD(device->rtc),
          dims
        };
        normalize_cdf_y->launch(device->rtc,1,1,&kernelArgs);
      }
      device->rtc->sync();
    }
  }
  
  
  bool EnvMapLight::set2i(const std::string &member,
                          const vec2i &value) 
  {
    return false;
  }

  bool EnvMapLight::set3f(const std::string &member,
                          const vec3f &value) 
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

  bool EnvMapLight::setObject(const std::string &member,
                              const Object::SP &value) 
  {
    if (member == "texture") {
      params.texture = value->as<Texture>();
      return true;
    }
    return false;
  }


  EnvMapLight::EnvMapLight(Context *context, int slot)
    : Light(context,slot)
  {
    // std::cout << OWL_TERMINAL_YELLOW
    //           << "#bn: created env-map light"
    //           << OWL_TERMINAL_DEFAULT << std::endl;
    auto rtc = getRTC();
    cdf_y
      = rtc->createBuffer(sizeof(float));
    allCDFs_x
      = rtc->createBuffer(sizeof(float));

    computeWeights_xy
      = rtc->createCompute("computeWeights_xy");
    computeCDFs_doLine
      = rtc->createCompute("computeCDFs_doLine");
    normalize_cdf_y
      = rtc->createCompute("normalize_cdf_y");
  }
  
}

RTC_CUDA_COMPUTE(computeWeights_xy,barney::ComputeWeights_xy);
RTC_CUDA_COMPUTE(computeCDFs_doLine,barney::ComputeCDFs_doLine);
RTC_CUDA_COMPUTE(normalize_cdf_y,barney::Normalize_cdf_y);
