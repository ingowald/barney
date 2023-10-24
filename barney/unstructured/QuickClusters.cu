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

#include "barney/unstructured/QuickClusters.h"
#include "barney/Context.h"
#include "hilbert.h"

namespace barney {

  extern "C" char QuickClusters_ptx[];
  
  OWLGeomType UMeshQC::createGeomType(DevGroup *devGroup)
  {
    std::cout << OWL_TERMINAL_GREEN
              << "creating 'UMeshQC' geometry type"
              << OWL_TERMINAL_DEFAULT << std::endl;
    
    static OWLVarDecl params[]
      = {
         { "vertices", OWL_BUFPTR, OWL_OFFSETOF(DD,vertices) },
         { "tetIndices", OWL_BUFPTR, OWL_OFFSETOF(DD,tetIndices) },
         { "hexIndices", OWL_BUFPTR, OWL_OFFSETOF(DD,hexIndices) },
         { "elements", OWL_BUFPTR, OWL_OFFSETOF(DD,elements) },
         { "clusters", OWL_BUFPTR, OWL_OFFSETOF(DD,clusters) },
         { "numElements", OWL_INT, OWL_OFFSETOF(DD,numElements) },
         { "xf.values", OWL_BUFPTR, OWL_OFFSETOF(DD,xf.values) },
         { "xf.domain", OWL_FLOAT2, OWL_OFFSETOF(DD,xf.domain) },
         { "xf.baseDensity", OWL_FLOAT, OWL_OFFSETOF(DD,xf.baseDensity) },
         { "xf.numValues", OWL_INT, OWL_OFFSETOF(DD,xf.numValues) },
         { nullptr }
    };
    OWLModule module = owlModuleCreate
      (devGroup->owl,QuickClusters_ptx);
    OWLGeomType gt = owlGeomTypeCreate
      (devGroup->owl,OWL_GEOM_USER,sizeof(UMeshQC::DD),
       params,-1);
    owlGeomTypeSetBoundsProg(gt,module,"UMeshQCBounds");
    owlGeomTypeSetIntersectProg(gt,/*ray type*/0,module,"UMeshQCIsec");
    owlGeomTypeSetClosestHit(gt,/*ray type*/0,module,"UMeshQCCH");
    owlBuildPrograms(devGroup->owl);
    
    return gt;
  }
  
  UMeshQC::UMeshQC(DataGroup *owner,
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
  {
  }
    
  uint64_t UMeshQC::encodeBox(const box4f &box4f)
  {
    box3f box((const vec3f&)box4f.lower,(const vec3f&)box4f.upper);
    
    int maxValue = (1<<numHilbertBits)-1;
    vec3f center = box.center();
    // PRINT(box);
    // PRINT(worldBounds);
    center
      = (center - getPos(worldBounds.lower))
      * rcp(max(vec3f(1e-10f),getPos(worldBounds.size())));
    // PRINT(center);
    center = clamp(center,vec3f(0.f),vec3f(1.f));
    vec3ul coords = vec3ul(center * maxValue);
    // PRINT(coords);
    bitmask_t _coords[3];
    _coords[0] = coords.x;
    _coords[1] = coords.y;
    _coords[2] = coords.z;
    return hilbert_c2i(3,numHilbertBits,_coords);
  }

  uint64_t UMeshQC::encodeTet(int primID)
  {
    const TetIndices indices = tetIndices[primID];
    return encodeBox(box4f()
                     .including(vertices[indices[0]])
                     .including(vertices[indices[1]])
                     .including(vertices[indices[2]])
                     .including(vertices[indices[3]]));
  }

  uint64_t UMeshQC::encodeHex(int primID)
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
    

  
  void UMeshQC::build(Volume *volume)
  {
    std::cout << "umesh: computing world bounds" << std::endl;
    worldBounds = box4f();
    for (int i=0;i<vertices.size();i++)
      worldBounds.extend((const vec4f&)vertices[i]);
    // PING; PRINT(worldBounds);

    std::cout << "umesh: building hilbert prims" << std::endl;
    std::vector<std::pair<uint64_t,uint32_t>> hilbertPrims
      (tetIndices.size()+hexIndices.size());
    owl::common::parallel_for_blocked
      (0,(int)tetIndices.size(),1024,
       [&](int begin, int end) {
         for (int i=begin;i<end;i++)
           hilbertPrims[i] = {encodeTet(i),(i<<3)|TET};
       });
    // for (int i=0;i<tetIndices.size();i++) 
    //   hilbertPrims[i] = {encodeTet(i),(i<<3)|TET};
    owl::common::parallel_for_blocked
      (0,(int)hexIndices.size(),1024,
       [&](int begin, int end) {
         for (int i=begin;i<end;i++)
           // for (int i=0;i<hexIndices.size();i++) 
           hilbertPrims[tetIndices.size()+i] = {encodeHex(i),(i<<3)|HEX};
       });
    std::cout << "umesh: sorting prims" << std::endl;
    std::sort(hilbertPrims.begin(),hilbertPrims.end());

    // for (int i=0;i<100;i++)
    //   PRINT((int*)hilbertPrims[i].first);
    
    std::vector<Element> elements;
    for (auto prim : hilbertPrims) {
      Element elt;
      elt.ID = prim.second >> 3;
      elt.type = prim.second & 0x7;
      elements.push_back(elt);
    }

    OWLGeomType gt = owner->devGroup->getOrCreateGeomTypeFor
      ("UMeshQC",UMeshQC::createGeomType);
    OWLGeom geom = owlGeomCreate(getOWL(),gt);
    owlGeomSet1i(geom,"numElements",(int)elements.size());

    PING; std::cout << "MEMORY LEAK!" << std::endl;
    OWLBuffer elementsBuffer
      = owlDeviceBufferCreate(getOWL(),
                              OWL_INT,
                              elements.size(),
                              elements.data());
    owlGeomSetBuffer(geom,"elements",elementsBuffer);
    
    PING; PRINT(worldBounds);
    PRINT(worldBounds.lower.w);
    PRINT(worldBounds.upper.w);
    if (volume->xf.domain.lower < volume->xf.domain.upper) {
      owlGeomSet2f(geom,"xf.domain",volume->xf.domain.lower,volume->xf.domain.upper);
    } else {
      owlGeomSet2f(geom,"xf.domain",worldBounds.lower.w,worldBounds.upper.w);
    }
    owlGeomSet1f(geom,"xf.baseDensity",volume->xf.baseDensity);
    owlGeomSet1i(geom,"xf.numValues",(int)volume->xf.values.size());

    OWLBuffer xfValuesBuffer = volume->xf.valuesBuffer;
    owlGeomSetBuffer(geom,"xf.values",xfValuesBuffer);
    
    PING; std::cout << "MEMORY LEAK!" << std::endl;
    OWLBuffer verticesBuffer
      = owlDeviceBufferCreate(getOWL(),
                              OWL_FLOAT4,
                              vertices.size(),
                              vertices.data());
    owlGeomSetBuffer(geom,"vertices",verticesBuffer);


    PING; std::cout << "MEMORY LEAK!" << std::endl;
    OWLBuffer tetIndicesBuffer
      = owlDeviceBufferCreate(getOWL(),
                              OWL_INT,
                              4*tetIndices.size(),
                              tetIndices.data());
    owlGeomSetBuffer(geom,"tetIndices",tetIndicesBuffer);
    
    OWLBuffer hexIndicesBuffer
      = owlDeviceBufferCreate(getOWL(),
                              OWL_INT,
                              8*hexIndices.size(),
                              hexIndices.data());
    owlGeomSetBuffer(geom,"hexIndices",hexIndicesBuffer);

    int numClusters = divRoundUp((int)elements.size(),clusterSize);

    PING; std::cout << "MEMORY LEAK!" << std::endl;
    OWLBuffer clustersBuffer
      = owlDeviceBufferCreate(getOWL(),
                              OWL_USER_TYPE(Cluster),
                              numClusters,nullptr);
    owlGeomSetBuffer(geom,"clusters",clustersBuffer);

    owlGeomSetPrimCount(geom,numClusters);
    
    volume->userGeoms.push_back(geom);
  }


  ScalarField *DataGroup::createUMesh(std::vector<vec4f> &vertices,
                                      std::vector<TetIndices> &tetIndices,
                                      std::vector<PyrIndices> &pyrIndices,
                                      std::vector<WedIndices> &wedIndices,
                                      std::vector<HexIndices> &hexIndices)
  {
    ScalarField::SP sf
      = std::make_shared<UMeshQC>(this,
                                  vertices,
                                   tetIndices,
                                   pyrIndices,
                                   wedIndices,
                                   hexIndices);
    
    return getContext()->initReference(sf);
  }
  

}
