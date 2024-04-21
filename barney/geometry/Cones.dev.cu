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

#include "barney/geometry/Cones.h"
#include "owl/owl_device.h"

namespace barney {
  
  OPTIX_BOUNDS_PROGRAM(ConesBounds)(const void *geomData,
                                        owl::common::box3f &bounds,  
                                        const int32_t primID)
  {
    auto &self = *(Cones::DD*)geomData;
    const vec2i pidx
      = self.indices
      ? self.indices[primID]
      : (2 * primID + vec2i(0, 1));
    
    const auto pa = self.vertices[pidx.x];
    const auto pb = self.vertices[pidx.y];
    
    const float ra = self.radii[pidx.x];
    const float rb = self.radii[pidx.y];
    box3f aBox(pa-ra,pa+ra);
    box3f bBox(pb-ra,pb+rb);
    
    bounds.lower = min(aBox.lower,bBox.lower);
    bounds.upper = max(aBox.upper,bBox.upper);
  }

  OPTIX_CLOSEST_HIT_PROGRAM(ConesCH)()
  {
    /* nothign - already set in isec */
  }

  inline __device__ float sqrt(float f) { return sqrtf(f); }
  inline __device__ float inversesqrt(float f) { return 1./sqrtf(f); }
  inline __device__ float length2(vec3f v) { return dot(v,v); }
  
  /*! largely stolen from VisRTX */
  OPTIX_INTERSECT_PROGRAM(ConesIsec)()
  {
    auto &ray = owl::getPRD<Ray>();
    auto &self = owl::getProgramData<Cones::DD>();
    int primID = optixGetPrimitiveIndex();
    
    float t_hit = optixGetRayTmax();

    vec3f ro = optixGetObjectRayOrigin();
    vec3f rd = optixGetObjectRayDirection();
    
    const vec2i pidx
      = self.indices
      ? self.indices[primID]
      : (2 * primID + vec2i(0, 1));
    
      const auto p0 = self.vertices[pidx.x];
      const auto p1 = self.vertices[pidx.y];

      const float ra = self.radii[pidx.x];
      const float rb = self.radii[pidx.y];

      const vec3f ba = p1 - p0;
      const vec3f oa = ro - p0;
      const vec3f ob = ro - p1;

      const float m0 = dot(ba, ba);
      const float m1 = dot(oa, ba);
      const float m2 = dot(ob, ba);
      const float m3 = dot(rd, ba);

      if (m1 < 0.0f) {
        if (length2(oa * m3 - rd * m1) < (ra * ra * m3 * m3)) {
          // reportIntersection(-m1 / m3, -ba * inversesqrt(m0), 0.f);
          float t = -m1 / m3;
          vec3f N = normalize(-ba * inversesqrt(m0));
        vec3f P = (vec3f)optixGetWorldRayOrigin()+t*(vec3f)optixGetWorldRayDirection();
          // float u = 0.f;
          vec3f geometryColor(NAN,NAN,NAN);
          ray.setHit(P,N,t,self.material,vec2f(NAN),geometryColor);
          optixReportIntersection(t,0);
        }
      } else if (m2 > 0.0f) {
        if (length2(ob * m3 - rd * m2) < (rb * rb * m3 * m3)) {
          // reportIntersection(-m2 / m3, ba * inversesqrt(m0), 1.f);
          float t = -m2 / m3;
          vec3f N = normalize(ba * inversesqrt(m0));
        vec3f P = (vec3f)optixGetWorldRayOrigin()+t*(vec3f)optixGetWorldRayDirection();
          // float u = 1.f;
          vec3f geometryColor(NAN,NAN,NAN);
          ray.setHit(P,N,t,self.material,vec2f(NAN),geometryColor);
          optixReportIntersection(t,0);
        }
      }

      const float m4 = dot(rd, oa);
      const float m5 = dot(oa, oa);
      const float rr = ra - rb;
      const float hy = m0 + rr * rr;

      float k2 = m0 * m0 - m3 * m3 * hy;
      float k1 = m0 * m0 * m4 - m1 * m3 * hy + m0 * ra * (rr * m3 * 1.0f);
      float k0 = m0 * m0 * m5 - m1 * m1 * hy + m0 * ra * (rr * m1 * 2.0f - m0 * ra);

      const float h = k1 * k1 - k2 * k0;
      if (h < 0.0f)
        return;

      const float t = (-k1 - sqrt(h)) / k2;
      const float y = m1 + t * m3;
      if (y > 0.0f && y < m0) {
        vec3f N = normalize(m0 * (m0 * (oa + t * rd) + rr * ba * ra) - ba * hy * y);
        vec3f P = (vec3f)optixGetWorldRayOrigin()+t*(vec3f)optixGetWorldRayDirection();
        // float u = y / m0;
        vec3f geometryColor(NAN,NAN,NAN);
        ray.setHit(P,N,t,self.material,vec2f(NAN),geometryColor);
        optixReportIntersection(t,0);
      }
  }
}

