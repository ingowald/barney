// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "barney/ModelSlot.h"
/* all routines for point-element sampling/intersection - shold
   logically be part of this file, but kept in separate file because
   these were mostly imported from openvkl */
#include "barney/umesh/common/ElementIntersection.h"
#include "barney/volume/MCAccelerator.h"

namespace BARNEY_NS {

  enum {
    _ANARI_TET = 0,
    _ANARI_HEX = 1,
    _ANARI_PRISM = 2,
    _ANARI_PYR = 3
  };
  enum {
    _VTK_TET = 10,
    _VTK_HEX = 12,
    _VTK_PRISM = 13,
    _VTK_PYR = 14,
    _VTK_POLYHEDRON = 42
  };

  /*! scalar field represented by a unstructured mesh. can contain
      tets, pyramids, tents, or hexes */
  struct UMeshField : public ScalarField
  {
    typedef std::shared_ptr<UMeshField> SP;

    UMeshField(Context *context,
               const DevGroup::SP &devices);

    virtual ~UMeshField() = default;
    
    /*! helper class for representing an N-long integer tuple, to
       represent prism, pyramid, hex, etc elemnet indices */
    template<int N>
    struct ints { int v[N];
      inline __rtc_device int &operator[](int i)      { return v[i]; }
      inline __rtc_device int operator[](int i) const { return v[i]; }
    };
    /*! device-data for a unstructured-mesh scalar field, containing
        all device-side pointers and function to access this field and
        sample/evaluate its elemnets */
    struct DD : public ScalarField::DD {
      
      inline __rtc_device box4f cellBounds(uint32_t cellIdx) const;

      /* compute scalar of given umesh element at point P, and return
         that in 'retVal'. returns true if P is inside the elemnt,
         false if outside (in which case retVal is not defined) */
      inline __rtc_device bool eltScalar(float &retVal,
                                         uint32_t cellIdx,
                                         vec3f P,
                                         bool dbg = false) const;
      
      /* compute scalar of given tet in umesh, at point P, and return
         that in 'retVal'. returns true if P is inside the elemnt,
         false if outside (in which case retVal is not defined) */
      inline __rtc_device bool tetScalar(float &retVal,
                                         uint32_t cellIdx,
                                         vec3f P,
                                         bool dbg = false) const;
      
      /* compute scalar of given pyramid in umesh, at point P, and
         return that in 'retVal'. returns true if P is inside the
         elemnt, false if outside (in which case retVal is not
         defined) */
      inline __rtc_device bool pyrScalar(float &retVal,
                                         uint32_t cellIdx,
                                         vec3f P) const;
      
      /* compute scalar of given prism in umesh, at point P, and return
         that in 'retVal'. returns true if P is inside the elemnt,
         false if outside (in which case retVal is not defined) */
      inline __rtc_device bool prismScalar(float &retVal,
                                         uint32_t cellIdx,
                                         vec3f P) const;
      
      /* compute scalar of given hex in umesh, at point P, and return
         that in 'retVal'. returns true if P is inside the elemnt,
         false if outside (in which case retVal is not defined) */
      inline __rtc_device
      bool hexScalar(float &retVal,
                     uint32_t cellIdx,
                     vec3f P, bool
                     dbg=false) const;

      /* compute scalar of given polyhedron in umesh, at point P, and
         return that in 'retVal'. returns true if P is inside the
         element, false if outside (in which case retVal is not
         defined). Containment uses +X ray parity over face fans (convex /
         well-behaved cells; see `umesh_poly_stream`). Per-vertex scalars use
         inverse-distance weights over face-stream corners. */
      inline __rtc_device bool polyScalar(float &retVal,
                                          uint32_t cellIdx,
                                          vec3f P) const;

      const vec3f       *vertices;
      const float       *scalars;
      const int         *indices;
      const int         *cellOffsets;
      const uint8_t     *cellTypes;
      int                numCells;
      int                numVertices;
      int                numScalars;
      int                numIndices;
      bool               scalarsArePerVertex;
    };

    /*! build *initial* macro-cell grid (ie, the scalar field min/max
      ranges, but not yet the majorants) over a umesh */
    void buildInitialMacroCells(MCGrid &grid);

    /*! computes, on specified device, the bounding boxes and - if
      d_primRanges is non-null - the primitmives ranges. d_primBounds
      and d_primRanges (if non-null) must be pre-allocated and
      writeaable on specified device */
    void computeElementBBs(Device  *device,
                           box3f   *d_primBounds,
                           range1f *d_primRanges=0);
    
    // ------------------------------------------------------------------
    /*! @{ parameter set/commit interface */
    void commit() override;
    bool setData(const std::string &member,
                 const std::shared_ptr<Data> &value) override;
    /*! @} */
    // ------------------------------------------------------------------

    
    DD getDD(Device *device);

    /*! create, fill, and return a macrocell grid for this field */
    MCGrid::SP buildMCs() override;

    VolumeAccel::SP createAccel(Volume *volume) override;
    
    /*! creates an acceleration structure for a 'isoSurface' geometry
        using this scalar field type */
    IsoSurfaceAccel::SP createIsoAccel(IsoSurface *isoSurface) override;

    /*! returns part of the string used to find the optix device
        programs that operate on this type */
    static std::string typeName() { return "UMesh"; };

    /*! @{ set by the user, as paramters */
    PODData::SP scalars;
    PODData::SP indices;
    PODData::SP cellOffsets;
    PODData::SP/*uint8_t*/ cellTypes;
    PODData::SP vertices;
    int numCells;
    bool scalarsArePerVertex = false;
    /*! @} */
    struct PLD {
      box3f   *pWorldBounds = 0;
    };
    PLD *getPLD(Device *device);
    std::vector<PLD> perLogical;
  };

  /*! VTK_POLYHEDRON face stream: numFaces, then per face (n, i0..i_{n-1}).
      All index reads stay within numIndices; faces/points capped; each
      vertex index is validated before any vertices[]/scalars[] load.

      Containment (`polyScalar`) uses +X ray parity over triangle fans; that
      matches convex, consistently oriented polyhedra (typical CFD meshes).
      Non-convex cells can classify inside/outside incorrectly. */
  namespace umesh_poly_stream {
    static constexpr int kMaxFaces = 8192;
    static constexpr int kMaxPtsPerFace = 1024;

    struct FaceStreamReader {
      const int *indices;
      int numIndices;
      int numVertices;
      int numScalars;
      bool scalarsArePerVertex;
      uint32_t cellIdx;
      int pos;
      int numFaces;
      bool ok;

      inline __rtc_device FaceStreamReader(const int *indices_,
                                           int numIndices_,
                                           int numVertices_,
                                           int numScalars_,
                                           bool scalarsArePerVertex_,
                                           uint32_t cellIdx_,
                                           int offset)
          : indices(indices_),
            numIndices(numIndices_),
            numVertices(numVertices_),
            numScalars(numScalars_),
            scalarsArePerVertex(scalarsArePerVertex_),
            cellIdx(cellIdx_),
            pos(0),
            numFaces(0),
            ok(false)
      {
        if (offset < 0 || offset >= numIndices)
          return;
        pos = offset;
        if (pos + 1 > numIndices)
          return;
        numFaces = indices[pos++];
        if (numFaces <= 0 || numFaces > kMaxFaces)
          return;
        if (!scalarsArePerVertex) {
          const int sci = (int)cellIdx;
          if (sci < 0 || sci >= numScalars)
            return;
        }
        ok = true;
      }

      inline __rtc_device bool faceBegin(int &numPts)
      {
        if (!ok)
          return false;
        if (pos + 1 > numIndices) {
          ok = false;
          return false;
        }
        numPts = indices[pos++];
        if (numPts < 3 || numPts > kMaxPtsPerFace) {
          ok = false;
          return false;
        }
        if (pos + numPts > numIndices) {
          ok = false;
          return false;
        }
        return true;
      }

      inline __rtc_device bool readVertex(int &vtxIdx, int &scalarIdx)
      {
        if (!ok || pos >= numIndices) {
          ok = false;
          return false;
        }
        vtxIdx = indices[pos++];
        if (vtxIdx < 0 || vtxIdx >= numVertices) {
          ok = false;
          return false;
        }
        scalarIdx = scalarsArePerVertex ? vtxIdx : (int)cellIdx;
        if (scalarsArePerVertex && (scalarIdx < 0 || scalarIdx >= numScalars)) {
          ok = false;
          return false;
        }
        return true;
      }
    };
  } // namespace umesh_poly_stream

  // ==================================================================
  // IMPLEMENTATION
  // ==================================================================

  inline __rtc_device
  box4f UMeshField::DD::cellBounds(uint32_t cellIdx) const
  {
    uint32_t cellType = cellTypes[cellIdx];
    int offset = cellOffsets[cellIdx];

    if (cellType == _VTK_POLYHEDRON) {
      umesh_poly_stream::FaceStreamReader stream(
          indices,
          numIndices,
          numVertices,
          numScalars,
          scalarsArePerVertex,
          cellIdx,
          offset);
      if (!stream.ok)
        return box4f{};
      box4f bb;
      for (int f = 0; f < stream.numFaces; f++) {
        int numPts = 0;
        if (!stream.faceBegin(numPts))
          return box4f{};
        for (int p = 0; p < numPts; p++) {
          int vtxIdx = 0, si = 0;
          if (!stream.readVertex(vtxIdx, si))
            return box4f{};
          bb.extend(vec4f(vertices[vtxIdx], scalars[si]));
        }
      }
      return bb;
    }

    uint32_t nv = 0;
    switch (cellType) {
    case _VTK_TET:
    case _ANARI_TET:
      nv = 4;
      break;
    case _VTK_PYR:
    case _ANARI_PYR:
      nv = 5;
      break;
    case _VTK_PRISM:
    case _ANARI_PRISM:
      nv = 6;
      break;
    case _VTK_HEX:
    case _ANARI_HEX:
      nv = 8;
      break;
    default:
      ;
    }

    box4f bb;
    if (nv == 0)
      return bb;
    if (offset < 0
        || (long long)offset + (long long)nv > (long long)numIndices)
      return bb;
    const int sci = scalarsArePerVertex ? 0 : (int)cellIdx;
    if (!scalarsArePerVertex && (sci < 0 || sci >= numScalars))
      return bb;

    for (uint32_t i = 0; i < nv; i++) {
      int vtxIdx = indices[offset++];
      if (vtxIdx < 0 || vtxIdx >= numVertices)
        return box4f{};
      const int si = scalarsArePerVertex ? vtxIdx : sci;
      if (scalarsArePerVertex && (si < 0 || si >= numScalars))
        return box4f{};
      vec4f v(vertices[vtxIdx], scalars[si]);
      bb.extend(v);
    }
    return bb;
  }

  /*! evaluate (relative) distance of point P to the implicit plane
      defined by points A,B,C. distance is not normalized */
  inline __rtc_device
  float evalToImplicitPlane(vec3f P, vec3f a, vec3f b, vec3f c)
  {
    vec3f N = cross(b-a,c-a);
    return dot(P-a,N);
  }

  /*! evaluate (relative) distance of point P to the implicit plane
      defined by points A,B,C. distance is not normalized */
  inline __rtc_device
  float evalToImplicitPlane(vec3f P, vec4f a, vec4f b, vec4f c)
  { return evalToImplicitPlane(P,getPos(a),getPos(b),getPos(c)); }

  /* compute scalar of given umesh element at point P, and return that
     in 'retVal'. returns true if P is inside the elemnt, false if
     outside (in which case retVal is not defined) */
  inline __rtc_device
  bool UMeshField::DD::eltScalar(float &retVal,
                                 uint32_t cellIdx,
                                 vec3f P,
                                 bool dbg) const
  {
    uint8_t cellType = cellTypes[cellIdx];
    switch (cellType) {
    case _ANARI_TET:
    case _VTK_TET:
      return tetScalar(retVal,cellIdx,P,dbg);
    case _ANARI_PYR:
    case _VTK_PYR:
      return pyrScalar(retVal,cellIdx,P);
    case _ANARI_PRISM:
    case _VTK_PRISM:
      return prismScalar(retVal,cellIdx,P);
    case _ANARI_HEX:
    case _VTK_HEX:
      return hexScalar(retVal,cellIdx,P,dbg);
    case _VTK_POLYHEDRON:
      return polyScalar(retVal,cellIdx,P);
    }
    return false;
  }
  
  inline __rtc_device
  bool UMeshField::DD::tetScalar(float &retVal,
                                 uint32_t cellIdx,
                                 vec3f P,
                                 bool dbg) const
  {
    int ofs0 = cellOffsets[cellIdx];
    if (ofs0 < 0 || (long long)ofs0 + 4 > (long long)numIndices)
      return false;
    const int *ix = this->indices + ofs0;
    vec4i indices(ix[0], ix[1], ix[2], ix[3]);
    auto badVtx = [&](int v) { return v < 0 || v >= numVertices; };
    if (badVtx(indices.x) || badVtx(indices.y) || badVtx(indices.z)
        || badVtx(indices.w))
      return false;
    if (scalarsArePerVertex) {
      auto badS = [&](int s) { return s < 0 || s >= numScalars; };
      if (badS(indices.x) || badS(indices.y) || badS(indices.z)
          || badS(indices.w))
        return false;
    } else if ((int)cellIdx >= numScalars) {
      return false;
    }

    vec4f v0(vertices[indices.x],scalars[scalarsArePerVertex?indices.x:cellIdx]);
    vec4f v1(vertices[indices.y],scalars[scalarsArePerVertex?indices.y:cellIdx]);
    vec4f v2(vertices[indices.z],scalars[scalarsArePerVertex?indices.z:cellIdx]);
    vec4f v3(vertices[indices.w],scalars[scalarsArePerVertex?indices.w:cellIdx]);

    float t3 = evalToImplicitPlane(P,v0,v1,v2);
    if (t3 < 0.f) return false;
    float t2 = evalToImplicitPlane(P,v0,v3,v1);
    if (t2 < 0.f) return false;
    float t1 = evalToImplicitPlane(P,v0,v2,v3);
    if (t1 < 0.f) return false;
    float t0 = evalToImplicitPlane(P,v1,v3,v2);
    if (t0 < 0.f) return false;

    if (scalarsArePerVertex) {
      float scale = 1.f/(t0+t1+t2+t3);
      retVal = scale * (t0*v0.w + t1*v1.w + t2*v2.w + t3*v3.w);
    } else {
      retVal = scalars[cellIdx];
    }
    return true;
  }
  
  inline __rtc_device
  bool UMeshField::DD::pyrScalar(float &retVal,
                                 uint32_t cellIdx,
                                 vec3f P) const
  {
    int ofs0 = cellOffsets[cellIdx];
    if (ofs0 < 0 || (long long)ofs0 + 5 > (long long)numIndices)
      return false;
    UMeshField::ints<5> indices;
    const int *src = this->indices + ofs0;
    for (int k = 0; k < 5; k++)
      indices[k] = src[k];
    for (int k = 0; k < 5; k++) {
      int v = indices[k];
      if (v < 0 || v >= numVertices) return false;
      if (scalarsArePerVertex && (v < 0 || v >= numScalars)) return false;
    }
    if (!scalarsArePerVertex && (int)cellIdx >= numScalars) return false;
    vec4f v0(vertices[indices[0]],scalars[scalarsArePerVertex?indices[0]:cellIdx]);
    vec4f v1(vertices[indices[1]],scalars[scalarsArePerVertex?indices[1]:cellIdx]);
    vec4f v2(vertices[indices[2]],scalars[scalarsArePerVertex?indices[2]:cellIdx]);
    vec4f v3(vertices[indices[3]],scalars[scalarsArePerVertex?indices[3]:cellIdx]);
    vec4f v4(vertices[indices[4]],scalars[scalarsArePerVertex?indices[4]:cellIdx]);
    return intersectPyrEXT(retVal, P, v0,v1,v2,v3,v4);
  }

  inline __rtc_device
  bool UMeshField::DD::prismScalar(float &retVal,
                                 uint32_t cellIdx,
                                 vec3f P) const
  {
    int ofs0 = cellOffsets[cellIdx];
    if (ofs0 < 0 || (long long)ofs0 + 6 > (long long)numIndices)
      return false;
    UMeshField::ints<6> indices;
    const int *src = this->indices + ofs0;
    for (int k = 0; k < 6; k++)
      indices[k] = src[k];
    for (int k = 0; k < 6; k++) {
      int v = indices[k];
      if (v < 0 || v >= numVertices) return false;
      if (scalarsArePerVertex && (v < 0 || v >= numScalars)) return false;
    }
    if (!scalarsArePerVertex && (int)cellIdx >= numScalars) return false;
    vec4f v0(vertices[indices[0]],scalars[scalarsArePerVertex?indices[0]:cellIdx]);
    vec4f v1(vertices[indices[1]],scalars[scalarsArePerVertex?indices[1]:cellIdx]);
    vec4f v2(vertices[indices[2]],scalars[scalarsArePerVertex?indices[2]:cellIdx]);
    vec4f v3(vertices[indices[3]],scalars[scalarsArePerVertex?indices[3]:cellIdx]);
    vec4f v4(vertices[indices[4]],scalars[scalarsArePerVertex?indices[4]:cellIdx]);
    vec4f v5(vertices[indices[5]],scalars[scalarsArePerVertex?indices[5]:cellIdx]);
    return intersectPrismEXT(retVal, P, v0,v1,v2,v3,v4,v5);
  }

  inline __rtc_device
  bool UMeshField::DD::hexScalar(float &retVal,
                                 uint32_t cellIdx,
                                 vec3f P,
                                 bool dbg) const
  {
    int ofs0 = cellOffsets[cellIdx];
    if (ofs0 < 0 || (long long)ofs0 + 8 > (long long)numIndices)
      return false;
    UMeshField::ints<8> indices;
    const int *src = this->indices + ofs0;
    for (int k = 0; k < 8; k++)
      indices[k] = src[k];
    for (int k = 0; k < 8; k++) {
      int v = indices[k];
      if (v < 0 || v >= numVertices) return false;
      if (scalarsArePerVertex && (v < 0 || v >= numScalars)) return false;
    }
    if (!scalarsArePerVertex && (int)cellIdx >= numScalars) return false;
    vec4f v0(vertices[indices[0]],scalars[scalarsArePerVertex?indices[0]:cellIdx]);
    vec4f v1(vertices[indices[1]],scalars[scalarsArePerVertex?indices[1]:cellIdx]);
    vec4f v2(vertices[indices[2]],scalars[scalarsArePerVertex?indices[2]:cellIdx]);
    vec4f v3(vertices[indices[3]],scalars[scalarsArePerVertex?indices[3]:cellIdx]);
    vec4f v4(vertices[indices[4]],scalars[scalarsArePerVertex?indices[4]:cellIdx]);
    vec4f v5(vertices[indices[5]],scalars[scalarsArePerVertex?indices[5]:cellIdx]);
    vec4f v6(vertices[indices[6]],scalars[scalarsArePerVertex?indices[6]:cellIdx]);
    vec4f v7(vertices[indices[7]],scalars[scalarsArePerVertex?indices[7]:cellIdx]);
    return intersectHexEXT(retVal, P, v0,v1,v2,v3,v4,v5,v6,v7, dbg);
  }

  inline __rtc_device
  bool UMeshField::DD::polyScalar(float &retVal,
                                  uint32_t cellIdx,
                                  vec3f P) const
  {
    const int offset = cellOffsets[cellIdx];
    umesh_poly_stream::FaceStreamReader stream(indices,
                                               numIndices,
                                               numVertices,
                                               numScalars,
                                               scalarsArePerVertex,
                                               cellIdx,
                                               offset);
    if (!stream.ok)
      return false;

    // ---- point-in-polyhedron via ray casting along +X ----
    int crossings = 0;
    for (int f = 0; f < stream.numFaces; f++) {
      int numPts = 0;
      if (!stream.faceBegin(numPts))
        return false;
      int v0Idx = 0, si0 = 0;
      if (!stream.readVertex(v0Idx, si0))
        return false;
      (void)si0;
      int v1Idx = 0, si1 = 0;
      if (!stream.readVertex(v1Idx, si1))
        return false;
      (void)si1;
      vec3f a = vertices[v0Idx];
      for (int t = 0; t < numPts - 2; t++) {
        int v2Idx = 0, si2 = 0;
        if (!stream.readVertex(v2Idx, si2))
          return false;
        (void)si2;
        vec3f b = vertices[v1Idx];
        vec3f c = vertices[v2Idx];
        // Moller-Trumbore ray-triangle intersection (ray: P + t*(1,0,0))
        vec3f e1 = b - a;
        vec3f e2 = c - a;
        // d x e2, where d = (1,0,0)
        vec3f h = vec3f(0.f, -e2.z, e2.y);
        float det = dot(e1, h);
        if (fabsf(det) < 1e-12f) {
          v1Idx = v2Idx;
          continue;
        }
        float inv_det = 1.f / det;
        vec3f s = P - a;
        float u = inv_det * dot(s, h);
        if (u < 0.f || u > 1.f) {
          v1Idx = v2Idx;
          continue;
        }
        vec3f q = cross(s, e1);
        float v = inv_det * q.x; // dot((1,0,0), q)
        if (v < 0.f || u + v > 1.f) {
          v1Idx = v2Idx;
          continue;
        }
        float ray_t = inv_det * dot(e2, q);
        if (ray_t > 0.f)
          crossings++;
        v1Idx = v2Idx;
      }
    }

    if ((crossings & 1) == 0)
      return false;

    // ---- scalar evaluation ----
    if (!scalarsArePerVertex) {
      retVal = scalars[cellIdx];
      return true;
    }

    umesh_poly_stream::FaceStreamReader idwStream(indices,
                                                numIndices,
                                                numVertices,
                                                numScalars,
                                                scalarsArePerVertex,
                                                cellIdx,
                                                offset);
    if (!idwStream.ok)
      return false;
    float weightSum = 0.f;
    float valueSum = 0.f;
    for (int f = 0; f < idwStream.numFaces; f++) {
      int numPts = 0;
      if (!idwStream.faceBegin(numPts))
        return false;
      for (int p = 0; p < numPts; p++) {
        int vtxIdx = 0, si = 0;
        if (!idwStream.readVertex(vtxIdx, si))
          return false;
        vec3f vp = vertices[vtxIdx];
        float dist2 = dot(vp - P, vp - P);
        float w = 1.f / fmaxf(dist2, 1e-20f);
        weightSum += w;
        valueSum += w * scalars[si];
      }
    }
    if (weightSum <= 0.f)
      return false;
    retVal = valueSum / weightSum;
    return true;
  }

}
