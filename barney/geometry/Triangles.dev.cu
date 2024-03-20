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

#include "barney/geometry/Triangles.h"
#include <owl/owl_device.h>

namespace barney {
 
  __device__
  inline vec4f getAttribute(const Triangles::DD &self,
                            const vec3i triangle,
                            int attr, float u, float v)
  {
    vec4f result{0.f, 0.f, 0.f, 1.f};

    const vec4f *colors{nullptr};
    //if (self.vertexAttribute[attr]) { // TODO: primitive attributes
    //  colors = self.vertexAttribute[attr];
    //}
    if (self.vertexAttribute[attr]) {
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

  OPTIX_CLOSEST_HIT_PROGRAM(TrianglesCH)()
  {
    auto &ray = owl::getPRD<Ray>();
    auto &self = owl::getProgramData<Triangles::DD>();
    int primID = optixGetPrimitiveIndex();
    vec3i triangle = self.indices[primID];
    vec3f v0 = self.vertices[triangle.x];
    vec3f v1 = self.vertices[triangle.y];
    vec3f v2 = self.vertices[triangle.z];
    vec3f n = cross(v1-v0,v2-v0);
    n = optixTransformNormalFromObjectToWorldSpace(n);
    n = normalize(n);
    
    vec3f dir = optixGetWorldRayDirection();
    // auto mat = self.material;

    const float u = optixGetTriangleBarycentrics().x;
    const float v = optixGetTriangleBarycentrics().y;


    // ------------------------------------------------------------------
    // get texture coordinates
    // ------------------------------------------------------------------
    vec2f tc(u,v);
    if (self.texcoords) {
      const vec2f Ta = self.texcoords[triangle.x];
      const vec2f Tb = self.texcoords[triangle.y];
      const vec2f Tc = self.texcoords[triangle.z];
      tc = ((1.f-u-v)*Ta + u*Tb + v*Tc);
    }
#if VISUALIZE_PRIMS
    colorFromTexture /*mat.baseColor*/ *= owl::randomColor(primID);
#endif

#if 1
    vec3f geometryColor(NAN);
    if (self.material.materialType == render::MATTE) {
      if (self.material.matte.samplerType == render::IMAGE1D) {
        int attr = self.material.matte.sampler.image1D.inAttribute;
        mat4f inTransform = self.material.matte.sampler.image1D.inTransform;
        vec4f inOffset = self.material.matte.sampler.image1D.inOffset;
        mat4f outTransform = self.material.matte.sampler.image1D.outTransform;
        vec4f outOffset = self.material.matte.sampler.image1D.outOffset;
        const vec4f *image = self.material.matte.sampler.image1D.image.data;
        int imageWidth = self.material.matte.sampler.image1D.image.width;

        vec4f inAttr = getAttribute(self,triangle,attr,u,v);

        inAttr = inTransform * inAttr + inOffset;

        float f = clamp(inAttr.x,0.f,1.f); // TODO: other wrap modes!

        int i = min(int(f*imageWidth),imageWidth-1);

        vec4f sample = image[i]; // TODO: linear/nearest interpolation (textures?!)

        sample = outTransform * sample + outOffset;

        geometryColor = vec3f(sample);
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

        vec4f inAttr = getAttribute(self,triangle,attr,u,v);

        inAttr = inTransform * inAttr + inOffset;

        float f1 = clamp(inAttr.x,0.f,1.f); // TODO: other wrap modes!
        float f2 = clamp(inAttr.y,0.f,1.f); // TODO: other wrap modes!

        int x = min(int(f1*imageWidth),imageWidth-1);
        int y = min(int(f2*imageHeight),imageHeight-1);

        vec4f sample = image[x+imageWidth*y]; // TODO: linear/nearest interpolation (textures?!)

        sample = outTransform * sample + outOffset;

        geometryColor = vec3f(sample);
      }
      else if (self.material.matte.samplerType == render::TRANSFORM) {
        int attr = self.material.matte.sampler.transform.inAttribute;
        mat4f outTransform = self.material.matte.sampler.transform.outTransform;
        vec4f outOffset = self.material.matte.sampler.transform.outOffset;
        //printf("%f,%f,%f,%f\n",outOffset.x,outOffset.y,outOffset.z,outOffset.w);

        vec4f inAttr = getAttribute(self,triangle,attr,u,v);
        geometryColor = vec3f(outTransform * inAttr + outOffset);
      }
    }
#endif
    
    const vec3f osP  = (1.f-u-v)*v0 + u*v1 + v*v2;
    vec3f P  = optixTransformPointFromObjectToWorldSpace(osP);
    ray.setHit(P,n,optixGetRayTmax(),
               self.material,tc,geometryColor);
  }



  /*! triangles geom AH program; mostly check on transparency */
  OPTIX_ANY_HIT_PROGRAM(TrianglesAH)()
  {
    auto &ray  = owl::getPRD<Ray>();
    auto &self = owl::getProgramData<Triangles::DD>();

    if (!self.material.hasAlpha(ray.isShadowRay))
      return;
    
    int primID = optixGetPrimitiveIndex();
    vec3i triangle = self.indices[primID];
    
    const float u = optixGetTriangleBarycentrics().x;
    const float v = optixGetTriangleBarycentrics().y;

    // ------------------------------------------------------------------
    // get texture coordinates
    // ------------------------------------------------------------------
    vec2f tc(u,v);
    if (self.texcoords) {
      const vec2f Ta = self.texcoords[triangle.x];
      const vec2f Tb = self.texcoords[triangle.y];
      const vec2f Tc = self.texcoords[triangle.z];
      tc = ((1.f-u-v)*Ta + u*Tb + v*Tc);
    }

    float alpha = self.material.getAlpha(tc,ray.isShadowRay);
    if (alpha < 1.f && ((Random &)ray.rngSeed)() < 1.f-alpha) {
      optixIgnoreIntersection();
      return;
    }
  }
  
}
