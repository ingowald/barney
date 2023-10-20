// ======================================================================== //
// Copyright 2023-2023 Ingo Wald                                            //
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

#include "barney/Geometry.h"
#include "barney/DataGroup.h"
#include "cuBQL/bvh.h"
#include "hilbert.h"

namespace barney {

  struct UMeshGeometry : public Geometry {
    UMeshGeometry(DataGroup *owner,
                  const Material &material,
                  const vec4f *vertices, int numVertices,
                  const int *tets,       int numTets,
                  const int *pyrs,       int numPyrs,
                  const int *wedges,     int numWedges,
                  const int *hexes,      int numHexes
                  // ,
                  // const int *gridDescs,
                  // const int *gridIndices,
                  // int numGrids
                  )
      : Geometry(owner,material),
        tets(tets), numTets(numTets),
        pyrs(pyrs), numPyrs(numPyrs),
        wedges(wedges), numWedges(numWedges),
        hexes(hexes), numHexes(numHexes)
    {}

    const vec4f *vertices;
    int numVertices;
    const int *tets;   int numTets;
    const int *pyrs;   int numPyrs;
    const int *wedges; int numWedges;
    const int *hexes;  int numHexes;
  };
  
  struct UMesh_QC : public UMeshGeometry {
    enum { numHilbertBits = 20 };
    enum { TET=0, HEX } PrimType;
    UMesh_QC(DataGroup *owner,
             const Material &material,
             const vec4f *vertices, int numVertices,
             const int *tets,       int numTets,
             const int *pyrs,       int numPyrs,
             const int *wedges,     int numWedges,
             const int *hexes,      int numHexes)
      : UMeshGeometry(owner,material,
                      vertices,  numVertices,
                      tets,      numTets,
                      pyrs,      numPyrs,
                      wedges,    numWedges,
                      hexes,     numHexes)
    {}

    uint64_t encodeBox(const box4f &box4f)
    {
      box3f box((const vec3f&)box4f.lower,(const vec3f&)box4f.upper);
      int maxValue = (1<<numHilbertBits)-1;
      vec3f center = box.center();
      center = (center - box.lower) * rcp(min(vec3f(1e-10f),box.size()));
      center = clamp(center,vec3f(0.f),vec3f(1.f));
      vec3ul coords = vec3ul(center * maxValue);
      return hilbert_c2i(3,numHilbertBits,(bitmask_t*)&coords.x);
    }
    uint64_t encodeTet(int primID)
    {
      const int *indices = this->tets + 3*primID;
      return encodeBox(box4f()
                       .including(vertices[indices[0]])
                       .including(vertices[indices[1]])
                       .including(vertices[indices[2]]));
    }
    uint64_t encodeHex(int primID)
    {
      const int *indices = this->hexes + 8*primID;
      return encodeBox(box4f()
                       .including(vertices[indices[0]])
                       .including(vertices[indices[1]])
                       .including(vertices[indices[2]])
                       .including(vertices[indices[3]])
                       .including(vertices[indices[4]])
                       .including(vertices[indices[5]])
                       .including(vertices[indices[6]])
                       .including(vertices[indices[7]]));
    }
    
    /*! pretty-printer for printf-debugging */
    std::string toString() const override
    { return "UMesh(via quick clusters){}"; }

    void build() override;

    box3f worldBounds;
  };

  void UMesh_QC::build()
  {
    worldBounds = box3f();
    for (int i=0;i<numVertices;i++)
      worldBounds.extend((const vec3f&)vertices[i]);
    std::vector<std::pair<uint64_t,uint32_t>> hilbertPrims;
    for (int i=0;i<numTets;i++) 
      hilbertPrims.push_back({encodeTet(i),(i<<3)|TET});
    for (int i=0;i<numHexes;i++) 
      hilbertPrims.push_back({encodeHex(i),(i<<3)|TET});
    std::sort(hilbertPrims.begin(),hilbertPrims.end());
  }

}
