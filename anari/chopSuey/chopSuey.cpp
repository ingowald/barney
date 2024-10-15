#include <algorithm>
#include <fstream>
#include <iostream>
#include <vector>
#include <float.h>
#include "chopSuey.h"

# define TERMINAL_RED "\033[0;31m"
# define TERMINAL_RESET "\033[0m"
# define TERMINAL_DEFAULT TERMINAL_RESET

namespace chop {

  struct {
    std::string inFileName = "";
    std::string outFileName = "chopSuey.tri";
    unsigned  numClusters = 1;
    Strategy strategy = Strategy::Median;
    struct {
      int3 dims{0};
      int bpc{1};
    } volume;
  } cmdline;

  void usage(const std::string &err)
  {
    if (err != "")
      std::cout << TERMINAL_RED << "\nFatal error: " << err
                << TERMINAL_DEFAULT << std::endl << std::endl;

    std::cout << "Usage: ./chopSuey inFile.obj -o outFile.tri -n numClusters" << std::endl;
    std::cout << std::endl;
    exit(1);
  }

  inline std::string getExt(const std::string &fileName)
  {
    int pos = fileName.rfind('.');
    if (pos == fileName.npos)
      return "";
    return fileName.substr(pos);
  }

  /* Mesh splitter. Meshes may only contain *one* geom! */
  MeshSplitter::MeshSplitter(unsigned numClusters, const Mesh::SP& mesh, const box3 bounds)
      : numClustersDesired(numClusters)
      , mesh(mesh)
      , modelBounds(bounds)
  {
    if (!modelBounds.valid()) {
      if (mesh->bounds.valid()) {
        modelBounds = mesh->bounds;
      }
    }

    if (!modelBounds.valid()) {
      for (auto &geom : mesh->geoms) {
        for (auto &idx : geom->index) {
          modelBounds.extend(geom->vertex[idx.x]);
          modelBounds.extend(geom->vertex[idx.y]);
          modelBounds.extend(geom->vertex[idx.z]);
        }
      }
    }

    Domain domain;
    domain.first = 0;
    domain.last  = mesh->geoms[0]->index.size();
    domain.bounds = modelBounds;

    clusters.push_back(domain);

    if (strategy != Strategy::Middle) {

      primRefs.resize(mesh->geoms[0]->index.size());

      for (size_t i=0; i<mesh->geoms[0]->index.size(); ++i) {
        int3 idx = mesh->geoms[0]->index[i];
        box3 primBounds = { float3(1e30f), float3(-1e30f) };
        primBounds.extend(mesh->geoms[0]->vertex[idx.x]);
        primBounds.extend(mesh->geoms[0]->vertex[idx.y]);
        primBounds.extend(mesh->geoms[0]->vertex[idx.z]);
        primRefs[i] = {(unsigned)i,primBounds.center()};
      }
    }

    if (numClustersDesired > 1)
      doSplit();

    for (auto d : clusters) {
      std::cout << d.first << ' ' << d.last << ' ' << d.bounds << '\n';
    }
  }

  void MeshSplitter::doSplit() {
    Domain domain;

    unsigned clusterToPick = 0;

    if (strategy == Strategy::Middle) {
      // Pick cluster with highest volume
      unsigned maxVolumeIndex = 0;
      float maxVolume = 0.f;
      for (unsigned i=0; i<clusters.size(); ++i)
      {
        if (clusters[i].bounds.volume() > maxVolume) {
          maxVolumeIndex = i;
          maxVolume = clusters[i].bounds.volume();
        }
      }
      clusterToPick = maxVolumeIndex;
    } else if (strategy == Strategy::Median) {
      // Pick cluster with most prims
      unsigned maxNumPrimsIndex = 0;
      unsigned maxNumPrims = 0;
      for (unsigned i=0; i<clusters.size(); ++i)
      {
        if (clusters[i].last-clusters[i].first > maxNumPrims) {
          maxNumPrimsIndex = i;
          maxNumPrims = clusters[i].last-clusters[i].first;
        }
      }
      clusterToPick = maxNumPrimsIndex;
    }

    domain = clusters[clusterToPick];
    clusters.erase(clusters.begin()+clusterToPick);

    int splitAxis = 0;
    if (domain.bounds.size()[1]>domain.bounds.size()[0]
     && domain.bounds.size()[1]>=domain.bounds.size()[2]) {
      splitAxis = 1;
    } else if (domain.bounds.size()[2]>domain.bounds.size()[0]
            && domain.bounds.size()[2]>=domain.bounds.size()[1]) {
      splitAxis = 2;
    }

    float splitPlane = FLT_MAX;

    if (strategy == Strategy::Middle)
      splitPlane = domain.bounds.lower[splitAxis]+domain.bounds.size()[splitAxis]*.5f;
    else if (strategy == Strategy::Median) {
      std::sort(primRefs.begin()+domain.first,
                primRefs.begin()+domain.last,
                [splitAxis](const PrimRef a, const PrimRef b) {
                  return a.centroid[splitAxis] < b.centroid[splitAxis];
                });
      size_t num = size_t(primRefs.size()/float(numClustersDesired));
      splitPlane = primRefs[domain.first+num].centroid[splitAxis];
    }

    std::partition(mesh->geoms[0]->index.begin()+domain.first,
                   mesh->geoms[0]->index.begin()+domain.last,
                   [&](int3 idx) {
                      const float3 &v1(mesh->geoms[0]->vertex[idx.x]);
                      const float3 &v2(mesh->geoms[0]->vertex[idx.y]);
                      const float3 &v3(mesh->geoms[0]->vertex[idx.z]);

                      float l = fminf(v1[splitAxis],fminf(v2[splitAxis],v3[splitAxis]));

                      return l<splitPlane;
                   });

    // Find first primitive where l >= splitPlane
    unsigned splitIndex = domain.first;
    for (unsigned i=domain.first; i<domain.last; ++i) {
      int3 idx = mesh->geoms[0]->index[i];
      const float3 &v1(mesh->geoms[0]->vertex[idx.x]);
      const float3 &v2(mesh->geoms[0]->vertex[idx.y]);
      const float3 &v3(mesh->geoms[0]->vertex[idx.z]);

      float l = fminf(v1[splitAxis],fminf(v2[splitAxis],v3[splitAxis]));

      if (l >= splitPlane) {
        splitIndex = i;
        break;
      }
    }

    Domain L;
    L.first  = domain.first;
    L.last   = splitIndex;
    L.bounds = domain.bounds;
    L.bounds.upper[splitAxis] = splitPlane;

    Domain R;
    R.first  = splitIndex;
    R.last   = domain.last;
    R.bounds = domain.bounds;
    R.bounds.lower[splitAxis] = splitPlane;

    if (L.last-L.first > 0)
      clusters.push_back(L);

    if (R.last-R.first > 0)
      clusters.push_back(R);

    if (clusters.size() < numClustersDesired)
      doSplit();
  }

  /* Volume splitter. Splits volumes into raw files, with domain and cellRange headers */
  VolumeSplitter::VolumeSplitter(unsigned numClusters, const Volume::SP& volume)
    : numClustersDesired(numClusters)
    , volume(volume)
  {
    Domain domain;
    domain.cellRange = box3i{{0,0,0},volume->dims};
    domain.voxelRange = box3i{{0,0,0},int3(volume->dims+1)};
    domain.spaceRange = box3{{0.f,0.f,0.f},float3(volume->dims)};

    cellRange = domain.cellRange;
    voxelRange = domain.voxelRange;
    spaceRange = domain.spaceRange;

    clusters.push_back(domain);

    doSplit();

    for (auto d : clusters) {
      std::cout << d.cellRange << ' ' << d.voxelRange << ' ' << d.spaceRange << '\n';
    }
  }

  void VolumeSplitter::doSplit() {
    Domain domain;

    unsigned clusterToPick = 0;

    // Pick cluster with highest volume
    unsigned maxVolumeIndex = 0;
    float maxVolume = 0.f;
    for (unsigned i=0; i<clusters.size(); ++i)
    {
      if (clusters[i].spaceRange.volume() > maxVolume) {
        maxVolumeIndex = i;
        maxVolume = clusters[i].spaceRange.volume();
      }
    }
    clusterToPick = maxVolumeIndex;

    domain = clusters[clusterToPick];
    clusters.erase(clusters.begin()+clusterToPick);

    int splitAxis = 0;
    if (domain.cellRange.size()[1]>domain.cellRange.size()[0]
     && domain.cellRange.size()[1]>=domain.cellRange.size()[2]) {
      splitAxis = 1;
    } else if (domain.cellRange.size()[2]>domain.cellRange.size()[0]
            && domain.cellRange.size()[2]>=domain.cellRange.size()[1]) {
      splitAxis = 2;
    }

    // Middle split
    int splitPlane = domain.cellRange.lower[splitAxis]+domain.cellRange.size()[splitAxis]*.5f;

    Domain L;
    L.cellRange = domain.cellRange;
    L.cellRange.upper[splitAxis] = splitPlane;
    L.voxelRange.lower = L.cellRange.lower;
    L.voxelRange.upper = min(L.cellRange.upper+1,volume->dims);
    L.spaceRange.lower = float3(L.cellRange.lower);
    L.spaceRange.upper = float3(L.cellRange.upper);

    Domain R;
    R.cellRange = domain.cellRange;
    R.cellRange.lower[splitAxis] = splitPlane;
    R.voxelRange.lower = R.cellRange.lower;
    R.voxelRange.upper = min(R.cellRange.upper+1,volume->dims);
    R.spaceRange.lower = float3(R.cellRange.lower);
    R.spaceRange.upper = float3(R.cellRange.upper);

    if (L.cellRange.volume() > 0)
      clusters.push_back(L);

    if (R.cellRange.volume() > 0)
      clusters.push_back(R);

    if (clusters.size() < numClustersDesired)
      doSplit();
  }

#if 0
  extern "C" int main(int argc, char **argv)
  {
    for (int i=1;i<argc;i++) {
      const std::string arg = argv[i];
      if (arg[0] != '-') {
        cmdline.inFileName = arg;
      }
      else if (arg == "-o") {
        cmdline.outFileName = argv[++i];
      }
      else if (arg == "-n") {
        cmdline.numClusters = std::atoi(argv[++i]);
      }
      else if (arg == "-dims") {
        cmdline.volume.dims.x = std::stoi(argv[++i]);
        cmdline.volume.dims.y = std::stoi(argv[++i]);
        cmdline.volume.dims.z = std::stoi(argv[++i]);
      }
      else if (arg == "-bpc") {
        cmdline.volume.bpc = std::stoi(argv[++i]);
      }
      else if (arg == "-type") {
        const std::string type = argv[++i];
        if (type == "uint8")
          cmdline.volume.bpc = 1;
        else if (type == "uint16")
          cmdline.volume.bpc = 2;
        else if (type == "float")
          cmdline.volume.bpc = 4;
        else
          usage("wrong type '"+type+"'");
      }
      else
        usage("unknown cmdline arg '"+arg+"'");
    }

    if (cmdline.inFileName == "")
      usage("no filename specified");

    box3 modelBounds = { float3(1e30f), float3(-1e30f) };

    if (getExt(cmdline.inFileName)==".obj") {
      Mesh::SP objMesh;
      try {
        objMesh = Mesh::load(cmdline.inFileName);
        // Construct bounds
        for (std::size_t i=0; i<objMesh->geoms.size(); ++i)
        {
          const Geometry::SP &geom = objMesh->geoms[i];
          for (const auto &v : geom->vertex) {
             modelBounds.extend(v);
          }
        }
        // Push onto geometry stack
        for (std::size_t i=0; i<objMesh->geoms.size(); ++i)
        {
          Geometry::SP &geom = objMesh->geoms[i];
        }
      } catch (...) { std::cerr << "Cannot load..\n"; }

      MeshSplitter splitter(cmdline.numClusters,objMesh,modelBounds);

      splitter.saveTris(cmdline.outFileName);
    } else if (getExt(cmdline.inFileName) == ".raw") {
      if (cmdline.volume.dims == int3(0))
        usage("no input dimensions specified");
      else if (cmdline.volume.bpc == 0)
        usage("no data type/bpc specified");

      Volume::SP volume = std::make_shared<Volume>(cmdline.inFileName,
                                                   cmdline.volume.dims,
                                                   cmdline.volume.bpc);

      VolumeSplitter splitter(cmdline.numClusters,volume);

      splitter.saveVols(cmdline.outFileName);
    }

    return 0;
  }
#endif
}


