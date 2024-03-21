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

#include "barney/common/barney-common.h"
#include "barney/common/mat4.h"
#include "barney/material/device/Material.h"
#include "owl/common/math/vec.h"

namespace barney {

  // TODO: _really_ use CUDA textures here...!!!
  struct SWSampler
  {
    const vec4f *data;
    int W, H;
    render::WrapMode wrapMode1, wrapMode2;
  };

  inline __device__
  float wrap(float f, int N, render::WrapMode mode)
  {
    if (mode == render::MIRROR) {
      if ((int(floorf(f)) & 1) == 1) // if is odd!
        return float(N-1)/N - (f - floorf(f));
      else
        return f - floorf(f);
    } else if (mode == render::WRAP) {
      return f - floorf(f);
    } else { // CLAMP
      return clamp(f,0.f,1.f-1.f/N);
    }
  }

  inline __device__
  vec4f sample1D(const SWSampler &sampler, float f)
  {
    int i = min(int(wrap(f,sampler.W,sampler.wrapMode1)*sampler.W),sampler.W-1);
    return sampler.data[i];
  }

  inline __device__
  vec4f sample2D(const SWSampler &sampler, float f1, float f2)
  {
    int x = min(int(wrap(f1,sampler.W,sampler.wrapMode1)*sampler.W),sampler.W-1);
    int y = min(int(wrap(f2,sampler.H,sampler.wrapMode2)*sampler.H),sampler.H-1);
    return sampler.data[x+sampler.W*y];
  }

  // For spheres:
  template<typename DD>
  inline __device__
  vec4f getAttribute(const DD &self,
                     const int primID, const int sphere,
                     int attr, float u, float v)
  {
    vec4f result{0.f, 0.f, 0.f, 1.f};

    if (self.primitiveAttribute[attr]) {
      return self.primitiveAttribute[attr][primID];
    }

    if (self.vertexAttribute[attr]) {
      return self.vertexAttribute[attr][sphere];
    }

    return result;
  }

  // Triangle attributes
  template<typename DD>
  inline __device__
  vec4f getAttribute(const DD &self,
                     const int primID, const vec3i triangle,
                     int attr, float u, float v)
  {
    vec4f result{0.f, 0.f, 0.f, 1.f};

    const vec4f *colors{nullptr};
    if (self.primitiveAttribute[attr]) {
      return self.primitiveAttribute[attr][primID];
    }
    else if (self.vertexAttribute[attr]) {
      colors = self.vertexAttribute[attr];
    }

    if (colors) {
      vec4f source1 = colors[triangle.x];
      vec4f source2 = colors[triangle.y];
      vec4f source3 = colors[triangle.z];
      // barycentric lerp:
      vec4f s1 = source3 * v;
      vec4f s2 = source2 * u;
      vec4f s3 = source1 * (1.f-u-v);
      result = s1+s2+s3;
    }

    return result;
  }

  template<typename DD,typename Primitive>
  inline __device__
  vec4f getColor(
      const DD &self, const int primID, const Primitive &primitive, float u, float v)
  {
    if (self.material.materialType == render::MATTE) {
      if (self.material.matte.samplerType == render::IMAGE1D) {
        int attr = self.material.matte.sampler.image1D.inAttribute;
        mat4f inTransform = self.material.matte.sampler.image1D.inTransform;
        vec4f inOffset = self.material.matte.sampler.image1D.inOffset;
        mat4f outTransform = self.material.matte.sampler.image1D.outTransform;
        vec4f outOffset = self.material.matte.sampler.image1D.outOffset;
        const vec4f *image = self.material.matte.sampler.image1D.image.data;
        int imageWidth = self.material.matte.sampler.image1D.image.width;
        render::WrapMode wrapMode = self.material.matte.sampler.image1D.image.wrapMode;

        vec4f inAttr = getAttribute(self,primID,primitive,attr,u,v);

        inAttr = inTransform * inAttr + inOffset;

        SWSampler sampler{image,imageWidth,0,wrapMode,wrapMode};
        vec4f sample = sample1D(sampler,inAttr.x);

        sample = outTransform * sample + outOffset;

        return sample;
      }
      if (self.material.matte.samplerType == render::IMAGE2D) {
        int attr = self.material.matte.sampler.image2D.inAttribute;
        mat4f inTransform = self.material.matte.sampler.image2D.inTransform;
        vec4f inOffset = self.material.matte.sampler.image2D.inOffset;
        mat4f outTransform = self.material.matte.sampler.image2D.outTransform;
        vec4f outOffset = self.material.matte.sampler.image2D.outOffset;
        const vec4f *image = self.material.matte.sampler.image2D.image.data;
        int imageWidth = self.material.matte.sampler.image2D.image.width;
        int imageHeight = self.material.matte.sampler.image2D.image.height;
        render::WrapMode wrapMode1 = self.material.matte.sampler.image2D.image.wrapMode1;
        render::WrapMode wrapMode2 = self.material.matte.sampler.image2D.image.wrapMode2;

        vec4f inAttr = getAttribute(self,primID,primitive,attr,u,v);

        inAttr = inTransform * inAttr + inOffset;

        SWSampler sampler{image,imageWidth,imageHeight,wrapMode1,wrapMode2};
        vec4f sample = sample2D(sampler,inAttr.x,inAttr.y);

        sample = outTransform * sample + outOffset;

        return sample;
      }
      else if (self.material.matte.samplerType == render::TRANSFORM) {
        int attr = self.material.matte.sampler.transform.inAttribute;
        mat4f outTransform = self.material.matte.sampler.transform.outTransform;
        vec4f outOffset = self.material.matte.sampler.transform.outOffset;
        //printf("%f,%f,%f,%f\n",outOffset.x,outOffset.y,outOffset.z,outOffset.w);

        vec4f inAttr = getAttribute(self,primID,primitive,attr,u,v);
        return outTransform * inAttr + outOffset;
      }
    }

    return vec4f(NAN);
  }
}
