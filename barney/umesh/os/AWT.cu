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
#if BARNEY_HAVE_CUDA
# include <cuBQL/builder/cuda/wide_gpu_builder.h>
#endif

namespace BARNEY_NS {
  
  RTC_IMPORT_COMPUTE1D(copyNodes);
  RTC_IMPORT_COMPUTE1D(computeMajorants);


  // struct ClearMajorants {
  //   AWTNode       *awtNodes;
  //   RefitInfo     *parents,
  //   int            numNodes;

  //   template<typename CI>
  //   inline __both__ void run(const CI &ci)
  //   {
  //     int nodeID = ci.launchIndex().x;
  //     if (nodeID >= numNodes) return;

  //     parents[nodeID].numRefits = 0;
  //     auto &node = awtNodes[nodeID];
  //     for (int childID=0;childID<AWT_NODE_WIDTH;childID++) {
  //       auto &child = node.child[childID];
  //       child.majorant = 0.f;
  //     }
  //   }
  // };

  struct ComputeMajorants {
    AWTNode       *awtNodes;
    RefitInfo     *refitInfos;
    uint32_t      *primIDs;
    int            numNodes;
    UMeshField::DD mesh;
    TransferFunction::DD xf;

    template<typename CI>
    inline __both__ void run(const CI &ci)
    {
      int tid = ci.launchIndex().x;
      int nodeID = tid / AWT_NODE_WIDTH;
      if (nodeID >= numNodes) return;
      
      auto *node = &awtNodes[nodeID];
      int childID = tid % AWT_NODE_WIDTH;
      auto *child = &node->child[childID];
      if (!child->nodeRef.isLeaf()) return;

      range1f leafRange;
      for (int i=0;i<(int)child->nodeRef.count;i++) {
        Element elt = mesh.elements[primIDs[child->nodeRef.offset+i]];
        leafRange.extend(getRange(mesh.eltBounds(elt)));
      }
      float majorant = xf.majorant(leafRange);
      child->majorant = majorant;
      while (true) {
        int parentID = refitInfos[nodeID].parent;
        if (parentID == -1) break;

        int numNotDone
          = ci.atomicAdd(&refitInfos[nodeID].numNotDone,-1)-1;
// #if __CUDA_ARCH__
//         __threadfence();
// #else
//         __builtin_ia32_mfence();
// #endif
        if (numNotDone != 0)
          break;

        majorant = 0.f;
        int numValid = 0;
        for (int i=0;i<(int)AWT_NODE_WIDTH;i++) {
          auto &child = node->child[i];
          if (!child.nodeRef.valid()) continue;
          majorant = max(majorant,child.majorant);
          ++numValid;
        }
        refitInfos[nodeID].numNotDone = numValid;
        
        nodeID  = parentID / AWT_NODE_WIDTH;
        childID = parentID % AWT_NODE_WIDTH;
        node    = &awtNodes[nodeID];
        child   = &node->child[childID];
        child->majorant = majorant;

        
        continue;
      }
    }
  };

  struct CopyNodes {
    AWTNode *out_nodes;
    RefitInfo *out_infos;
    const typename cuBQL::WideBVH<float,3,AWT_NODE_WIDTH>::Node *in_nodes;
    int numNodes;
    // Element *out_elements;
    Element *in_elements;
    uint32_t *primIDs;

    template<typename CI>
    inline __both__
    void run(const CI &ci)
    {
      int tid = ci.launchIndex().x;
      if (tid >= numNodes) return;

      if (tid == 0) {
        out_infos[tid].parent = -1;
      }
      
      int numActive = 0;
      for (int cid=0;cid<AWT_NODE_WIDTH;cid++) {
        auto &out = out_nodes[tid].child[cid];
        auto &in  = in_nodes[tid].children[cid];
        out.bounds.lower = (vec3f&)in.bounds.lower;
        out.bounds.upper = (vec3f&)in.bounds.upper;
        if (in.valid) {
          out.nodeRef.offset = in.offset;
          out.nodeRef.count = in.count;
          if (!out.nodeRef.isLeaf()) {
            out_infos[out.nodeRef.offset].parent = tid * AWT_NODE_WIDTH + cid;
          } else {
            // for (int j=0;j<out.nodeRef.count;j++)
            //   out_elements[out.nodeRef.offset+j]
            //     = in_elements[primIDs[out.nodeRef.offset+j]];
          }
          numActive++;
        } else {
          out.bounds = box3f();
          out.nodeRef.offset = 0;
          out.nodeRef.count = 0;
        }
      }
      out_infos[tid].numNotDone = numActive;
    }
  };

  // rtc::GeomType *AWTAccel::createGeomType(rtc::Device *device,
  //                               const void *cbData)
  // {
  //   return device->createUserGeomType("AWT_ptx",
  //                                     "AWT",
  //                                     sizeof(DD),
  //                                     /*ah*/false,/*ch*/false);
  // }

  AWTAccel::AWTAccel(Volume *volume,
                     UMeshField *mesh)
    : VolumeAccel(volume),
      mesh(mesh)
  {
    perLogical.resize(devices->numLogical);
  }

  void AWTAccel::build(bool full_rebuild)
  {
#if BARNEY_HAVE_CUDA
    for (auto device : *devices) {
      auto pld = getPLD(device);
      UMeshField::PLD *meshPLD = mesh->getPLD(device);
      auto rtc = device->rtc;

      if (pld->copyNodes == 0) {
        pld->copyNodes        //= rtc->createCompute("copyNodes");
          = createCompute_copyNodes(rtc);
        pld->computeMajorants //= rtc->createCompute("computeMajorants");
          = createCompute_computeMajorants(rtc);
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
        // Element *tempElements
        //   = (Element*)rtc->allocMem(numElements*sizeof(Element));
        pld->refitInfos = (RefitInfo*)rtc->allocMem(numNodes*sizeof(RefitInfo));
        pld->awtNodes = (AWTNode *)rtc->allocMem(numNodes*sizeof(AWTNode));
        pld->numNodes = numNodes;
        {
          CopyNodes args = {
            pld->awtNodes,
            pld->refitInfos,
            bvh.nodes,
            pld->numNodes,
            // tempElements,
            meshPLD->elements,
            bvh.primIDs
          };
          int bs = 128;
          int nb = divRoundUp(numNodes,bs);
          pld->copyNodes->launch(nb,bs,&args);
        }
        device->sync();
        pld->primIDs = bvh.primIDs;
        bvh.primIDs = 0;
        cuBQL::cuda::free(bvh);

        // rtc->copy(meshPLD->elements,tempElements,numElements*sizeof(Element));
        rtc->sync();
        // rtc->freeMem(tempElements);

        rtc::GeomType *gt
          = device->geomTypes.get(createGeomType,
                                  this);
        pld->geom = gt->createGeom();
        pld->geom->setPrimCount(1);
        
        DD dd = getDD(device);
        pld->geom->setDD(&dd);
        // now put that into a instantiable group, and build it.
        pld->group = device->rtc->createUserGeomsGroup({pld->geom});
        pld->group->buildAccel();
        
        Volume::PLD *volumePLD = volume->getPLD(device);
        volumePLD->generatedGroups = { pld->group };
      }
      
      std::cout << "#AWT: re-computing majorants" << std::endl;
      {
        ComputeMajorants args = {
          pld->awtNodes,
          pld->refitInfos,
          pld->primIDs,
          pld->numNodes,
          mesh->getDD(device),
          this->volume->xf.getDD(device),
        };
        int bs = 128;
        int nb = divRoundUp(pld->numNodes*AWT_NODE_WIDTH,bs);
        pld->computeMajorants->launch(nb,bs,&args);
      }
      rtc->sync();
      std::cout << "#AWT: majorants built" << std::endl;

      // setting new dd; xf may have changed
      DD dd = getDD(device);
      pld->geom->setDD(&dd);
    }
#else
    throw std::runtime_error("umesh AWT backend currently requires CUDA");
#endif
  }

  AWTAccel::DD AWTAccel::getDD(Device *device)
  {
    auto pld = getPLD(device);
    DD dd;
    dd.mesh     = mesh->getDD(device);
    dd.awtNodes = pld->awtNodes;
    dd.xf       = this->volume->xf.getDD(device);
    dd.primIDs  = pld->primIDs;
    return dd;
  }

  RTC_EXPORT_COMPUTE1D(copyNodes,CopyNodes);
  RTC_EXPORT_COMPUTE1D(computeMajorants,ComputeMajorants);
}



