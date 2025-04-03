// ======================================================================== //
// Copyright 2023-2025 Ingo Wald                                            //
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

#pragma once

#include "rtcore/cudaCommon/Device.h"
#include <cuBQL/bvh.h>

namespace rtc {
  namespace cuda {

    using cuBQL::bvh3f;
    
    struct Device;
    
    struct Group {
      Group(Device *device);
      virtual ~Group();
      virtual void buildAccel() = 0;
      virtual void refitAccel() { buildAccel(); }
      rtc::device::AccelHandle getDD() const { return (rtc::device::AccelHandle)d_accel; }

      Device *const device;
      void *d_accel = 0;
      cuBQL::bvh3f::Node *bvhNodes = 0;
    };

    struct GeomGroup : public Group {
      struct Prim { int geomID; int primID; };

      struct DeviceRecord {
        // 0..8b
        uint8_t *sbt;
        // 8..16b
        cuBQL::bvh3f::Node *bvhNodes;
        // 16..24b
        Prim *prims;
        // 24..28b
        uint32_t sbtEntrySize;
        // 28b..32b
        bool isTrianglesGroup;
      };

      virtual DeviceRecord getRecord() = 0;
      GeomGroup(Device *device,
                const std::vector<Geom *> &geoms);
      const std::vector<Geom *> geoms;

      uint8_t *sbt          = 0;
      size_t   sbtEntrySize = 0;
      
      int    numPrims = 0;
      Prim  *prims    = 0;
    };

    struct InstanceGroup : public Group {
      struct InstanceRecord {
        affine3f worldToObjectXfm;
        affine3f objectToWorldXfm;
        GeomGroup::DeviceRecord group;
        uint32_t ID;
      };
      struct DeviceRecord {
        struct {
          const bvh3f::Node *nodes;
          const uint32_t    *primIDs;
        } bvh;
        InstanceRecord *instanceRecords;
      };
      
      InstanceGroup(Device *device,
                    const std::vector<Group *>  &groups,
                    const std::vector<int>      &instanceIDs,
                    const std::vector<affine3f> &xfms);
      void buildAccel() override;

      DeviceRecord   *d_deviceRecord = 0;
      InstanceRecord *d_instanceRecords = 0;
      bvh3f bvh = { 0,0,0,0 };
      const std::vector<Group *>  groups;
      const std::vector<int>      instanceIDs;
      const std::vector<affine3f> xfms;
    };
    
    struct TrianglesGeomGroup : public GeomGroup {
      TrianglesGeomGroup(Device *device, const std::vector<Geom *> &geoms);
      void buildAccel() override;
      DeviceRecord getRecord() override;
    };
    
    struct UserGeomGroup : public GeomGroup {
      UserGeomGroup(Device *device, const std::vector<Geom *> &geoms);
      void buildAccel() override;
      DeviceRecord getRecord() override;
    };
    
  }
}
