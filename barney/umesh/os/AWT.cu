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

#include "barney/umesh/os/AWT.h"
#include <cuBQL/bvh.h>
#include <cuBQL/builder/cuda/wide_gpu_builder.h>

namespace barney {

  struct SetLeafMajorants {
    AWTNode *awtNodes;
    int      numNodes;
    Element *elements;
  };

  struct PropagateMajorants {
    AWTNode  *awtNodes;
    int       numNodes;
    uint32_t *parents;
  };
  
  struct CopyNodes {
    AWTNode *out_nodes;
    uint32_t *parents;
    const typename cuBQL::WideBVH<float,3,AWT_NODE_WIDTH>::Node *in_nodes;
    int numNodes;
    Element *out_elements;
    Element *in_elements;
    uint32_t *primIDs;
  };

  AWTAccel::AWTAccel(Volume *volume,
                     UMeshField *mesh)
    : VolumeAccel(volume),
      mesh(mesh)
  {
    perLogical.resize(devices->numLogical);
  }

  void AWTAccel::build(bool full_rebuild)
  {
    for (auto device : *devices) {
      auto pld = getPLD(device);
      UMeshField::PLD *meshPLD = mesh->getPLD(device);
      auto rtc = device->rtc;

      if (pld->copyNodes == 0) {
        pld->copyNodes          = rtc->createCompute("copyNodes");
        pld->setLeafMajorants   = rtc->createCompute("setLeafMajorants");
        pld->propagateMajorants = rtc->createCompute("propagateMajorants");
      }
      
      if (pld->awtNodes == 0) {
        std::cout << "#AWT: building INITIAL BVH" << std::endl;
        cuBQL::WideBVH<float,3,AWT_NODE_WIDTH> bvh;
        int numElements = mesh->numElements;
        box3f *primBounds
          = (box3f*)device->rtc->allocMem(numElements*sizeof(box3f));
        range1f *valueRanges
          = (range1f*)device->rtc->allocMem(numElements*sizeof(range1f));
        mesh->computeElementBBs(device,
                                primBounds,valueRanges);
        rtc->sync();
        rtc->copy(&pld->bounds,meshPLD->pWorldBounds,sizeof(box3f));
        
        SetActiveGPU forDuration(device);
        cuBQL::BuildConfig buildConfig;
        buildConfig.maxAllowedLeafSize = AWTNode::max_leaf_size;
        buildConfig.makeLeafThreshold = AWTNode::max_leaf_size;
        cuBQL::gpuBuilder(bvh,
                          (const cuBQL::box_t<float,3>*)primBounds,
                          numElements,
                          buildConfig);
        rtc->sync();
        rtc->freeMem(primBounds);
        rtc->freeMem(valueRanges);

        int numNodes = bvh.numNodes;
        Element *tempElements
          = (Element*)rtc->allocMem(numElements*sizeof(Element));
        pld->parents  = (uint32_t*)rtc->allocMem(numNodes*sizeof(uint32_t));
        pld->awtNodes = (AWTNode *)rtc->allocMem(numNodes*sizeof(AWTNode));
        pld->numNodes = numNodes;

        {
          CopyNodes args = {
            pld->awtNodes,
            pld->parents,
            bvh.nodes,
            pld->numNodes,
            tempElements,
            meshPLD->elements,
            bvh.primIDs
          };
          int bs = 128;
          int nb = divRoundUp(numNodes,bs);
          pld->copyNodes->launch(nb,bs,&args);
        }
        cuBQL::cuda::free(bvh);

        rtc->copy(meshPLD->elements,tempElements,numElements*sizeof(Element));
        rtc->sync();
        rtc->freeMem(tempElements);
        
        rtc::GeomType *gt
          = device->geomTypes.get("UMeshAWT",
                                  createGeomType,
                                  this);
        pld->geom = gt->createGeom();
        pld->geom->setPrimCount(1);
        Volume::PLD *volumePLD = volume->getPLD(device);
        volumePLD->generatedGeoms = { pld->geom };
      }
      

      std::cout << "#AWT: re-computing majorants" << std::endl;
      {
        SetLeafMajorants args = {
          pld->awtNodes,
          pld->numNodes,
          mesh->getPLD(device)->elements,
        };
        int bs = 128;
        int nb = divRoundUp(pld->numNodes,bs);
        pld->setLeafMajorants->launch(nb,bs,&args);
      }
      {
        PropagateMajorants args = {
          pld->awtNodes,
          pld->numNodes,
          pld->parents
        };
        int bs = 128;
        int nb = divRoundUp(pld->numNodes,bs);
        pld->propagateMajorants->launch(nb,bs,&args);
      }
      rtc->sync();
      std::cout << "#AWT: majorants built" << std::endl;
      
      DD dd = getDD(device);
      pld->geom->setDD(&dd);
    }
  }

}

