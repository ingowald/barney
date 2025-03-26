#pragma once

#include "rtcore/embree/Device.h"
#include "rtcore/embree/GeomType.h"

namespace rtc {
  namespace embree {

    struct InstanceGroup;
    
    struct TraceInterface {
      void ignoreIntersection(); 
      void reportIntersection(float t, int i);
      void *getPRD() const;
      const void *getProgramData() const;
      const void *getLPData() const;
      vec3i getLaunchDims()  const;
      vec3i getLaunchIndex() const;
      vec2f getTriangleBarycentrics() const;
      int getPrimitiveIndex() const;
      int getGeometryIndex() const;
      int getRTCInstanceIndex() const;
      int getInstanceID() const;
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
      /*! 'linear' instance index in the list of instances we (as an
          rtcore device) were supplied with. _not_ necessarily the
          same as whatever instances the one-higher-up api layer (ie
          barney) bay be using, and certainly not the user instance
          IDs that anari have been specified on anari side */
      int       instIdx;
      vec3f     worldOrigin;
      vec3f     worldDirection;
      void           *prd;
      const int      *instIDs;
      const void     *geomData;
      const void     *lpData;
      const affine3f *objectToWorldXfm;
      const affine3f *worldToObjectXfm;
      RTCRay         *embreeRay;
      RTCHit         *embreeHit;
      InstanceGroup  *world;
    };



    inline void TraceInterface::ignoreIntersection() 
    { ignoreThisHit = true;  }

    inline void TraceInterface::reportIntersection(float t, int i)
    { isec_t = t;  }

    inline void *TraceInterface::getPRD() const
    { return prd; }

    inline const void *TraceInterface::getProgramData() const
    { return geomData; }

    inline const void *TraceInterface::getLPData() const
    { return lpData; }

    inline vec3i TraceInterface::getLaunchDims()  const
    { return launchDimensions; }

    inline vec3i TraceInterface::getLaunchIndex() const
    { return launchIndex; }

    inline vec2f TraceInterface::getTriangleBarycentrics() const
    { return triangleBarycentrics; }

    inline int TraceInterface::getPrimitiveIndex() const
    { return primID; }

    inline int TraceInterface::getRTCInstanceIndex() const
    { return instIdx; }

    inline int TraceInterface::getInstanceID() const
    { return instIDs[instIdx]; }

    inline int TraceInterface::getGeometryIndex() const
    { return geomID; }

    inline float TraceInterface::getRayTmax() const
    { return embreeRay->tfar; }

    inline float TraceInterface::getRayTmin() const
    { return embreeRay->tnear; }

    inline vec3f TraceInterface::getObjectRayDirection() const
    { return *(vec3f*)&embreeRay->dir_x; }

    inline vec3f TraceInterface::getObjectRayOrigin() const
    { return *(vec3f*)&embreeRay->org_x; }

    inline vec3f TraceInterface::getWorldRayDirection() const
    { return worldDirection; }

    inline vec3f TraceInterface::getWorldRayOrigin() const
    { return worldOrigin; }
  }
}
