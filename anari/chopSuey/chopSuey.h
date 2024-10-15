#pragma once

#include <algorithm>
#include <fstream>
#include <iostream>
#include <vector>
#include <float.h>
#include "mesh.h"
#include "volume.h"
#include "box1.h"
#include "box3.h"

# define TERMINAL_RED "\033[0;31m"
# define TERMINAL_RESET "\033[0m"
# define TERMINAL_DEFAULT TERMINAL_RESET

namespace chop {

  using float3 = anari::math::float3;
  using int3 = anari::math::int3;
  using box1 = anari::math::box1;
  using box3 = anari::math::box3;
  using box3i = anari::math::box3i;
  inline std::ostream &operator<<(std::ostream &out, float3 v)
  { out<<'('<<v.x<<','<<v.y<<','<<v.z<<')'; return out; }
  inline std::ostream &operator<<(std::ostream &out, int3 v)
  { out<<'('<<v.x<<','<<v.y<<','<<v.z<<')'; return out; }
  inline std::ostream &operator<<(std::ostream &out, box1 b)
  { out<<'['<<b.lower<<':'<<b.upper<<']'; return out; }
  inline std::ostream &operator<<(std::ostream &out, box3 b)
  { out<<'['<<b.lower<<':'<<b.upper<<']'; return out; }
  inline std::ostream &operator<<(std::ostream &out, box3i b)
  { out<<'['<<b.lower<<':'<<b.upper<<']'; return out; }

  enum class Strategy { Middle, Median, };

  /* Mesh splitter. Meshes may only contain *one* geom! */
  struct MeshSplitter {

    struct Domain {
      unsigned first, last;
      box3 bounds;
    };

    struct PrimRef {
      unsigned primID;
      float3 centroid;
    };

    MeshSplitter(unsigned numClusters, const Mesh::SP& mesh, const box3 bounds);

    void doSplit();

    Strategy strategy = Strategy::Middle;

    unsigned numClustersDesired;
    Mesh::SP mesh;
    box3 modelBounds;

    std::vector<Domain> clusters;

    std::vector<PrimRef> primRefs;
  };

  /* Volume splitter. Splits volumes into raw files, with domain and cellRange headers */
  struct VolumeSplitter {

    struct Domain {
      // Number of cells
      box3i cellRange;

      // Number of vertices of the dual grid
      box3i voxelRange;

      // Space covered
      box3 spaceRange;
    };

    VolumeSplitter(unsigned numClusters, const Volume::SP& volume);

    void doSplit();

    unsigned numClustersDesired;
    Volume::SP volume;
    Mesh::SP mesh;
    box3i cellRange;
    box3i voxelRange;
    box3  spaceRange;

    std::vector<Domain> clusters;
  };

}


