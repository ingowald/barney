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
#include "barney/common/math.h"

namespace BARNEY_NS {

  struct EnvMapLight : public Light {
    typedef std::shared_ptr<EnvMapLight> SP;
    EnvMapLight(Context *context,
                const DevGroup::SP &devices);

    struct DD {
#if RTC_DEVICE_CODE
      inline __rtc_device float pdf(vec3f dir, bool dbg=false) const;
      inline __rtc_device Light::Sample sample(Random &r, bool dbg=false) const;
      inline __rtc_device vec3f  eval(vec3f dir, bool dbg=false) const;
      /*! converts from a given pixel's coordinates into the
          world-space vector that poitns to the center of that
          pixel */
      inline __rtc_device vec3f  uvToWorld(float sx, float sy) const;
      inline __rtc_device vec3f  pixelToWorld(vec2i pixelID) const;
      inline __rtc_device vec2i  worldToPixel(vec3f worldDir) const;
#endif
      linear3f            toWorld;
      linear3f            toLocal;
      // cudaTextureObject_t texture;
      rtc::device::TextureObject texture = 0;
      vec2i               dims;
      const float        *cdf_y = 0;
      const float        *allCDFs_x = 0;
    };

    DD getDD(Device *device, const affine3f &xfm);

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
    Texture::SP texture;

    linear3f   toWorld;
    linear3f   toLocal;
    // OWLTexture texture = 0;
    // OWLBuffer  cdf_y;
    // OWLBuffer  allCDFs_x;

    struct PLD {
      rtc::Buffer  *cdf_y = 0;
      rtc::Buffer  *allCDFs_x = 0;
      
      rtc::ComputeKernel2D *computeWeights_xy;
      rtc::ComputeKernel1D *computeCDFs_doLine;
      rtc::ComputeKernel1D *normalize_cdf_y;
    };

    PLD *getPLD(Device *);
    std::vector<PLD> perLogical;
    vec2i      dims;
  };



#if RTC_DEVICE_CODE
  inline __rtc_device
  float cdfGetPDF(int position, const float *cdf, int N)
  {
    float f_at_position = cdf[position];
    float f_before_position
      = position
      ? cdf[position-1]
      : 0.f;
    
    return N*(f_at_position-f_before_position);
  }

  inline __rtc_device
  int sampleCDF(const float *cdf, int N, float v,
                float &pdf, bool dbg = false)
  {
    // if (dbg) printf("****** sampling sdf with %i items, v = %f\n",N,v);
    int begin = 0;
    int end   = N;
    while ((end - begin) > 1) {
      int mid = (begin+end)/2;
      float f_mid = cdf[mid-1];

      // if (dbg)
      //   printf("[%i %i] -> %i -> %f\n",
      //          begin,end,mid,f_mid);
      if (v < f_mid)
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
    
    pdf = N*(f_at_position-f_before_position);
    return position;
  }
  
  inline __rtc_device float
  EnvMapLight::DD::pdf(vec3f dir, bool dbg) const
  {
    if (!texture)
      // if we don't have a texture barney will use a env-map light
      // with constant ambient radiance; the pdf of that will be a
      // uniform 1/(4*PI). It would be cleaner to have this handled in
      // a wrapper class that handles both env-map and non-env-map
      // cases (instead of returning a proper non-envmap pdf from the
      // envmap class...), but shadeRays currently only asks the
      // envmap for the light pdf, so this is the best way to put
      // it... for now.
      return ONE_OVER_FOUR_PI;
    
    vec2i pixel = worldToPixel(dir);
    float pdf_y = cdfGetPDF(pixel.y,cdf_y,dims.y);

    float pdf_x = cdfGetPDF(pixel.x,allCDFs_x+pixel.y*dims.x,dims.x);

    float rel_y = (pixel.y+.5f) / dims.y;
    const float theta = ONE_PI * rel_y;
    // pdf *= (ONE_PI/sinf(theta));

    // return pdf;

    return pdf_x * pdf_y * 1.f/(TWO_PI*ONE_PI*sinf(theta));
  }
  
  inline __rtc_device float pbrt_clampf(float f, float lo, float hi)
  { return max(lo,min(hi,f)); }
  
  inline __rtc_device float pbrtSphericalTheta(const vec3f &v)
  {
    return acosf(pbrt_clampf(v.z, -1.f, 1.f));
  }
  
  inline __rtc_device float pbrtSphericalPhi(const vec3f &v)
  {
    float p = atan2f(v.y, v.x);
    return (p < 0.f) ? (p + float(2.f * M_PI)) : p;
  }

  inline __rtc_device vec2i
  EnvMapLight::DD::worldToPixel(vec3f worldDir) const
  {
    vec3f localDir = xfmVector(toLocal,worldDir);
    
    float theta = pbrtSphericalTheta(localDir);
    float phi   = pbrtSphericalPhi(localDir);
    const float invPi  = ONE_OVER_PI;
    const float inv2Pi = ONE_OVER_TWO_PI;
    vec2f uv(phi * inv2Pi, theta * invPi);
    
    int ix = (int)clamp(uv.x*(float)dims.x,0.f,dims.x-1.f);
    int iy = (int)clamp(uv.y*(float)dims.y,0.f,dims.y-1.f);

    return {ix,iy};
  }
  
  inline __rtc_device vec3f
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

  inline __rtc_device vec3f
  EnvMapLight::DD::uvToWorld(float f_x, float f_y) const
  {
    const float phi   = TWO_PI * f_x;
    const float theta = ONE_PI * f_y;
    
    vec3f dir;
    dir.z = cosf(theta);
    dir.x = cosf(phi)*sinf(theta);
    dir.y = sinf(phi)*sinf(theta);

    return xfmVector(toWorld,dir);
  }

  inline __rtc_device Light::Sample
  EnvMapLight::DD::sample(Random &r, bool dbg) const
  {
    if (!texture) return {};

    float r_y = r();
    float r_x = r();
    float pdf_y;
    // if (dbg) printf(" *** sampling cdf (w/ r=%f) in y\n",r_y);
    int iy = sampleCDF(cdf_y,dims.y,r_y,pdf_y,dbg);
    // if (dbg) printf("found iy %i, pdf %f/%f\n",iy,pdf_y,pdf_y*dims.y);
    float pdf_x;
    // if (dbg) printf(" *** sampling cdf (w/ r=%f)in x for y=%i\n",r_x,iy);
    int ix = sampleCDF(allCDFs_x+dims.x*iy,dims.x,r_x,pdf_x,dbg);
    // if (dbg) printf("found ix %i, pdf %f/%f\n",ix,pdf_x,pdf_x*dims.x);

#if 1
    float sx = (ix+r())/dims.x;
    float sy = (iy+r())/dims.y;
#else
    float sx = (ix+.5f)/dims.x;
    float sy = (iy+.5f)/dims.y;
#endif
    vec4f fromTex = rtc::tex2D<vec4f>(texture,sx,sy);
    Light::Sample sample;
    sample.radiance = (vec3f&)fromTex;
    sample.direction = uvToWorld(sx,sy);
    // sample.direction = pixelToWorld({ix,iy});
     // if (dbg) printf("found pixel %i %i -> maxrad %f, world %f %f %f\n",
     //                 ix,iy,
     //                 reduce_max(sample.radiance),
     //                 sample.direction.x,
     //                 sample.direction.y,
     //                 sample.direction.z);

    // pdf_x *= dims.x;
    // pdf_y *= dims.y;

    // *(dims.x*dims.y)
      // *ONE_OVER_FOUR_PI;
    // float rel_y = (iy+.5f) / dims.y;
    float rel_y = sy;
    const float theta = ONE_PI * rel_y;
    // sample.pdf *= (ONE_PI/sinf(theta));
    sample.pdf
      = pdf_x*pdf_y
      * 1.f/(TWO_PI*ONE_PI*sinf(theta));
    
    sample.distance = BARNEY_INF;
    return sample;
  }
#endif
}
