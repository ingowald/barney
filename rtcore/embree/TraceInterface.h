#pragma once

#include "rtcore/embree/Device.h"
#include "rtcore/embree/GeomType.h"

namespace rtc {
  namespace embree {

    struct InstanceGroup;
    
    struct TraceInterface {
      // #  ifdef __CUDA_ARCH__
      //       /* the embree compute interface only makes sense on the host,
      //          and for the device sometimes isn't even callable (std::atomic
      //          etc), so let's make those fcts 'go away' for device code */
      // #  else
      void ignoreIntersection(); 
      void reportIntersection(float t, int i);
      void *getPRD() const;
      const void *getProgramData() const;
      const void *getLPData() const;
      vec3i getLaunchDims()  const;
      vec3i getLaunchIndex() const;
      vec2f getTriangleBarycentrics() const;
      int getPrimitiveIndex() const;
      int getInstanceIndex() const;
      float getRayTmax() const;
      float getRayTmin() const;
      vec3f getObjectRayDirection() const;
      vec3f getObjectRayOrigin() const;
      vec3f getWorldRayDirection() const;
      vec3f getWorldRayOrigin() const;
      vec3f transformNormalFromObjectToWorldSpace(vec3f v) const;
      vec3f transformPointFromObjectToWorldSpace(vec3f v) const;
      vec3f transformVectorFromObjectToWorldSpace(vec3f v) const;
      vec3f transformNormalFromWorldToObjectSpace(vec3f v) const;
      vec3f transformPointFromWorldToObjectSpace(vec3f v) const;
      vec3f transformVectorFromWorldToObjectSpace(vec3f v) const;
      void  traceRay(rtc::device::AccelHandle world,
                     vec3f org,
                     vec3f dir,
                     float t0,
                     float t1,
                     void *prdPtr);

      /* this HAS to be the first entry! :*/
      RTCRayQueryContext embreeRayQueryContext;
      vec3i     launchIndex;
      vec3i     launchDimensions;
      bool      ignoreThisHit;
      float     isec_t;
      vec2f     triangleBarycentrics;
      int       primID;
      int       geomID;
      int       instID;
      vec3f     worldOrigin;
      vec3f     worldDirection;
      void           *prd;
      const void     *geomData;
      const void     *lpData;
      const affine3f *objectToWorldXfm;
      const affine3f *worldToObjectXfm;
      RTCRay         *embreeRay;
      RTCHit         *embreeHit;
      InstanceGroup  *world;
    };
  }
}
