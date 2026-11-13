// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "native/DeviceGroup.h"
#include "native/render/Sampler.h"
#include "native/render/DeviceWorld.h"

namespace BARNEY_NS {
  namespace native {

    // struct LDGContext;
    struct DeviceMaterial;
    struct HostMaterial;
    struct Group;
    
    /*! the rendering/path racing related part of a model that describes
      global render settings like light sources, background, envmap,
      etc

      iw - TODO probably want to mege this into LDGContext; this
      already has per-LDG sampler registry etc, so could/should also
      keep this data, too
    */
    struct World {
      typedef std::shared_ptr<World> SP;

      using DD = DeviceWorld;
      struct {
        EnvMapLight::SP light;
        affine3f        xfm;
      } envMapLight;

      World(Device *device);//LDGContext *ldgContext);
      virtual ~World();

      void set(const std::vector<QuadLight::DD> &quadLights);
      void set(const std::vector<DirLight::DD> &dirLights);
      void set(const std::vector<PointLight::DD> &pointLights);
      void set(EnvMapLight::SP envMapLight, const affine3f &xfm);
      void setMotionDeltas(const std::vector<affine3f> &deltas);
      void setInstanceOrientations(const std::vector<affine3f> &transforms);

      void setInstances(Group **groups,
                        const affine3f *xfms,
                        int numInstances);
      void updateInstanceTransforms(const affine3f *xfms, int numInstances);
      void setInstanceAttributes(const std::string &which, const PODData::SP &data);
      void setInstanceMotionDeltas(const affine3f *deltas, int numInstances);
      void updateWorldLightsFromInstances();
      void flattenInstancesForDevice(Device *device,
                                     std::vector<rtc::Group *> *rtcGroups,
                                     std::vector<affine3f> &rtcTransforms,
                                     std::vector<int> *inputInstIDs);

      struct {
        std::vector<std::shared_ptr<Group>> groups;
        std::vector<affine3f>  xfms;
        std::vector<int>       userIDs;
      } instances;
      rtc::Group *instanceGroup = 0;
      rtc::AccelHandle getInstanceAccel()
      {
        return rtc::getAccelHandle(instanceGroup);
      }
      
      PODData::SP instanceAttributes[5];
      PODData::SP instanceUserIDs;
      
      DD getDD();//Device *device);

      // struct PLD {
      struct {
        QuadLight::DD *quadLights = 0;
        int numQuadLights = 0;
        DirLight::DD *dirLights = 0;
        int numDirLights = 0;
        PointLight::DD *pointLights = 0;
        int numPointLights = 0;
        float *instanceOrientations = nullptr;
        affine3f *motionDeltas = 0;
        int numMotionDeltas = 0;
      } rtc;
      // };
      // PLD *getPLD(Device *device);
      
      // std::vector<PLD> perLogical;
      // DevGroup::SP const devices;
      // LDGContext *const ldgContext;
      Device *const device;
    };

  }
}
