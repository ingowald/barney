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

#include "barney/Volume.h"
#include "barney/DataGroup.h"
#include "cuBQL/bvh.h"
#include "hilbert.h"
// temp:
#include "barney/Context.h"

namespace barney {

  struct UMeshField : public ScalarField {
    typedef std::shared_ptr<UMeshField> SP;
    
    UMeshField(DataGroup *owner,
               std::vector<vec4f> &vertices,
               std::vector<TetIndices> &tetIndices,
               std::vector<PyrIndices> &pyrIndices,
               std::vector<WedIndices> &wedIndices,
               std::vector<HexIndices> &hexIndices)
      : ScalarField(owner),
        tetIndices(std::move(tetIndices)),
        pyrIndices(std::move(pyrIndices)),
        wedIndices(std::move(wedIndices)),
        hexIndices(std::move(hexIndices))
    {}

    std::vector<vec4f>      vertices;
    std::vector<TetIndices> tetIndices;
    std::vector<PyrIndices> pyrIndices;
    std::vector<WedIndices> wedIndices;
    std::vector<HexIndices> hexIndices;
  };
  
  struct UMesh_QC : public UMeshField {
    enum { numHilbertBits = 20 };
    enum { TET=0, HEX } PrimType;
    UMesh_QC(DataGroup *owner,
             std::vector<vec4f> &vertices,
             std::vector<TetIndices> &tetIndices,
             std::vector<PyrIndices> &pyrIndices,
             std::vector<WedIndices> &wedIndices,
             std::vector<HexIndices> &hexIndices)
      : UMeshField(owner,
                   vertices,
                   tetIndices,
                   pyrIndices,
                   wedIndices,
                   hexIndices)
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
      const TetIndices indices = tetIndices[primID];
      return encodeBox(box4f()
                       .including(vertices[indices[0]])
                       .including(vertices[indices[1]])
                       .including(vertices[indices[2]]));
    }
    uint64_t encodeHex(int primID)
    {
      const HexIndices indices = hexIndices[primID];
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

    void build(Volume *volume) override;

    box3f worldBounds;
  };

  void UMesh_QC::build(Volume *volume)
  {
    worldBounds = box3f();
    for (int i=0;i<vertices.size();i++)
      worldBounds.extend((const vec3f&)vertices[i]);
    std::vector<std::pair<uint64_t,uint32_t>> hilbertPrims;
    for (int i=0;i<tetIndices.size();i++) 
      hilbertPrims.push_back({encodeTet(i),(i<<3)|TET});
    for (int i=0;i<hexIndices.size();i++) 
      hilbertPrims.push_back({encodeHex(i),(i<<3)|TET});
    std::sort(hilbertPrims.begin(),hilbertPrims.end());
  }


  ScalarField *DataGroup::createUMesh(std::vector<vec4f> &vertices,
                                      std::vector<TetIndices> &tetIndices,
                                      std::vector<PyrIndices> &pyrIndices,
                                      std::vector<WedIndices> &wedIndices,
                                      std::vector<HexIndices> &hexIndices)
  {
    ScalarField::SP sf
      = std::make_shared<UMesh_QC>(this,
                                   vertices,
                                   tetIndices,
                                   pyrIndices,
                                   wedIndices,
                                   hexIndices);
    
    return getContext()->initReference(sf);
  }
  

}
