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

#pragma once

#include "barney/light/Light.h"
#include "barney/DeviceGroup.h"

namespace barney {

  struct EnvMapLight : public Light {
    typedef std::shared_ptr<EnvMapLight> SP;
    EnvMapLight(ModelSlot *owner);

    struct DD {
      inline __device__ float pdf(vec3f dir, bool dbg=false) const;
      inline __device__ Light::Sample sample(Random &r, bool dbg=false) const;
      inline __device__ vec3f  eval(vec3f dir, bool dbg=false) const;
      /*! converts from a given pixel's coordinates into the
          world-space vector that poitns to the center of that
          pixel */
      inline __device__ vec3f  pixelToWorld(vec2i pixelID) const;
      inline __device__ vec2i  worldToPixel(vec3f worldDir) const;
        
      linear3f            toWorld;
      linear3f            toLocal;
      cudaTextureObject_t texture;
      vec2i               dims;
      const float        *cdf_y;
      const float        *allCDFs_x;
    };

    DD getDD(const Device::SP &device) const;

    // ==================================================================
    std::string toString() const override { return "EnvMapLight"; }
    
    // ------------------------------------------------------------------
    /*! @{ parameter set/commit interface */
    void commit() override;
    bool set2i(const std::string &member, const vec2i &value) override;
    bool set3f(const std::string &member, const vec3f &value) override;
    // bool set4x3f(const std::string &member, const affine3f &value) override;
    bool setObject(const std::string &member, const Object::SP &value) override;
    /*! @} */
    // ------------------------------------------------------------------

  private:
    // helper class that computes the CDFs in x and y, for importance
    // sampling the map
    void computeCDFs();
  public:
    struct {
      vec3f       direction { 1.f, 0.f, 0.f };
      vec3f       up        { 0.f, 0.f, 1.f };
      Texture::SP texture;
    } params;

    linear3f   toWorld;
    linear3f   toLocal;
    OWLTexture texture = 0;
    OWLBuffer  cdf_y;
    OWLBuffer  allCDFs_x;
    vec2i      dims;
  };



#ifdef __CUDACC__
  inline __device__
  float cdfGetPDF(int position, const float *cdf, int N)
  {
    float f_at_position = cdf[position];
    float f_before_position
      = position
      ? cdf[position-1]
      : 0.f;
    
    return (f_at_position-f_before_position);
  }

  inline __device__
  int sampleCDF(const float *cdf, int N, float v,
                float &pdf, bool dbg = false)
  {
    if (dbg) printf("****** sampling sdf with %i items, v = %f\n",N,v);
    int begin = 0;
    int end   = N;
    while (end - begin > 1) {
      int mid = (begin+end)/2;
      float f_mid = cdf[mid];

      if (dbg)
        printf("[%i %i] -> %i -> %f\n",
               begin,end,mid,f_mid);
      if (v <= f_mid)
        end = mid;
      else
        begin = mid;
    }
    int position = begin;
    float f_at_position = cdf[begin];
    float f_before_position
      = begin
      ? cdf[begin-1]
      : 0.f;
    
    pdf = (f_at_position-f_before_position);
    if (dbg)
      printf("done, found idx %i, with query %f between %f and %f, pdf = %f\n",
             position,v,f_before_position,f_at_position,pdf);
    return position;
  }
  
  inline __device__ float
  EnvMapLight::DD::pdf(vec3f dir, bool dbg) const
  {
    if (!texture)
      return 0.f;
    
    vec2i pixel = worldToPixel(dir);
    float pdf_y = cdfGetPDF(pixel.y,cdf_y,dims.y);

    float pdf_x = cdfGetPDF(pixel.x,allCDFs_x+pixel.y*dims.x,dims.x);

    return pdf_x*pdf_y
      *(dims.x*dims.y)
      *ONE_OVER_FOUR_PI
      *(2.f/ONE_PI)
      ;
  }
  
  inline __device__ float pbrt_clampf(float f, float lo, float hi)
  { return max(lo,min(hi,f)); }
  
  inline __device__ float pbrtSphericalTheta(const vec3f &v)
  {
    return acosf(pbrt_clampf(v.z, -1.f, 1.f));
  }
  
  inline __device__ float pbrtSphericalPhi(const vec3f &v)
  {
    float p = atan2f(v.y, v.x);
    return (p < 0.f) ? (p + float(2.f * M_PI)) : p;
  }

  inline __device__ vec2i
  EnvMapLight::DD::worldToPixel(vec3f worldDir) const
  {
    vec3f localDir = xfmVector(toLocal,worldDir);
    
    float theta = pbrtSphericalTheta(localDir);
    float phi   = pbrtSphericalPhi(localDir);
    const float invPi  = ONE_OVER_PI;
    const float inv2Pi = ONE_OVER_TWO_PI;
    vec2f uv(phi * inv2Pi, theta * invPi);
    
    int ix = clamp(int(uv.x*dims.x),0,dims.x-1);
    int iy = clamp(int(uv.y*dims.y),0,dims.y-1);

    return {ix,iy};
  }
  
  inline __device__ vec3f
  EnvMapLight::DD::pixelToWorld(vec2i pixelID) const
  {
    const float f_x   = (pixelID.x+.5f)/dims.x;
    const float phi   = TWO_PI * f_x;
    const float f_y   = (pixelID.y+.5f)/dims.y;
    const float theta = ONE_PI * f_y;

    vec3f dir;
    dir.z = cosf(theta);
    dir.x = cosf(phi)*sinf(theta);
    dir.y = sinf(phi)*sinf(theta);

    return xfmVector(toWorld,dir);
  }

  inline __device__ Light::Sample
  EnvMapLight::DD::sample(Random &r, bool dbg) const
  {
    if (!texture) return {};
    
    float pdf_y;
    if (dbg) printf(" *** sampling cdf in y\n");
    int iy = sampleCDF(cdf_y,dims.y,r(),pdf_y,dbg);

    float pdf_x;
    if (dbg) printf(" *** sampling cdf in x for y=%i\n",iy);
    int ix = sampleCDF(allCDFs_x+dims.x*iy,dims.x,r(),pdf_x,dbg);

    float sx = (ix+.5f)/dims.x;
    float sy = (iy+.5f)/dims.y;
    float4 fromTex = tex2D<float4>(texture,sx,sy);
    Light::Sample sample;
    sample.radiance = (vec3f&)fromTex;
    sample.direction = pixelToWorld({ix,iy});
    if (dbg) printf("found pixel %i %i -> world %f %f %f\n",
                    ix,iy,
                    sample.direction.x,
                    sample.direction.y,
                    sample.direction.z);
    sample.pdf = pdf_x*pdf_y
      *(dims.x*dims.y)
      *ONE_OVER_FOUR_PI
      *(2.f/ONE_PI)
      ;
    sample.distance = INFINITY;
    return sample;
  }
// #else
//   /* dummy implementations for non-cuda compilers, just so we don't
//      get any 'undefined' warnings */
//   inline __device__ Light::Sample
//   EnvMapLight::DD::sample(Random &r, bool dbg) const { return {}; }
//   inline __device__ float
//   EnvMapLight::DD::pdf(vec3f dir, bool dbg) const { return 0.f; }
#endif
}