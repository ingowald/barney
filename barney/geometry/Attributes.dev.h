DEPRECATED

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
// #include "barney/material/device/Material.h"
#include "owl/common/math/vec.h"

namespace barney {

  // // For spheres:
  // template<typename DD>
  // inline __device__
  // vec4f getAttribute(const DD &self,
  //                    const int primID, const int sphere,
  //                    int attr, float u, float v)
  // {
  //   vec4f result{0.f, 0.f, 0.f, 1.f};

  //   if (self.primitiveAttribute[attr]) {
  //     return self.primitiveAttribute[attr][primID];
  //   }

  //   if (self.vertexAttribute[attr]) {
  //     return self.vertexAttribute[attr][sphere];
  //   }

  //   return result;
  // }

  // // Triangle attributes
  // template<typename DD>
  // inline __device__
  // vec4f getAttribute(const DD &self,
  //                    const int primID, const vec3i triangle,
  //                    int attr, float u, float v)
  // {
  //   vec4f result{0.f, 0.f, 0.f, 1.f};

  //   const vec4f *colors{nullptr};
  //   if (self.primitiveAttribute[attr]) {
  //     return self.primitiveAttribute[attr][primID];
  //   }
  //   else if (self.vertexAttribute[attr]) {
  //     colors = self.vertexAttribute[attr];
  //   }

  //   if (colors) {
  //     vec4f source1 = colors[triangle.x];
  //     vec4f source2 = colors[triangle.y];
  //     vec4f source3 = colors[triangle.z];
  //     // barycentric lerp:
  //     vec4f s1 = source3 * v;
  //     vec4f s2 = source2 * u;
  //     vec4f s3 = source1 * (1.f-u-v);
  //     result = s1+s2+s3;
  //   }

  //   return result;
  // }

  // template<typename DD,typename Primitive>
  // inline __device__
  // vec4f getColor(
  //     const DD &self, const int primID, const Primitive &primitive, float u, float v)
  // {
  //   if (self.material.materialType == render::MATTE) {
  //     if (self.material.matte.samplerType == render::IMAGE1D ||
  //         self.material.matte.samplerType == render::IMAGE2D) {
  //       int attr = self.material.matte.sampler.image.inAttribute;
  //       mat4f inTransform = self.material.matte.sampler.image.inTransform;
  //       vec4f inOffset = self.material.matte.sampler.image.inOffset;
  //       mat4f outTransform = self.material.matte.sampler.image.outTransform;
  //       vec4f outOffset = self.material.matte.sampler.image.outOffset;
  //       cudaTextureObject_t image = self.material.matte.sampler.image.image;

  //       vec4f inAttr = getAttribute(self,primID,primitive,attr,u,v);

  //       inAttr = inTransform * inAttr + inOffset;

  //       float yCoord
  //         = self.material.matte.samplerType == render::IMAGE1D ? 0.5f
  //                                                              : inAttr.y;
  //       vec4f sample = tex2D<float4>(image, inAttr.x, yCoord);

  //       sample = outTransform * sample + outOffset;

  //       return sample;
  //     }
  //     else if (self.material.matte.samplerType == render::TRANSFORM) {
  //       int attr = self.material.matte.sampler.transform.inAttribute;
  //       mat4f outTransform = self.material.matte.sampler.transform.outTransform;
  //       vec4f outOffset = self.material.matte.sampler.transform.outOffset;
  //       //printf("%f,%f,%f,%f\n",outOffset.x,outOffset.y,outOffset.z,outOffset.w);

  //       vec4f inAttr = getAttribute(self,primID,primitive,attr,u,v);
  //       return outTransform * inAttr + outOffset;
  //     }
  //   }

  //   return vec4f(NAN);
  // }
}
