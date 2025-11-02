// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#include "barney/render/World.h"
#include "barney/Context.h"
#include "barney/material/DeviceMaterial.h"
#include "barney/material/Material.h"
#include "barney/render/MaterialRegistry.h"
#include "barney/render/SamplerRegistry.h"

namespace BARNEY_NS {
  namespace render {

    World::World(SlotContext *slotContext)
      : devices(slotContext->devices),
        slotContext(slotContext)
    {
      perLogical.resize(devices->numLogical);
      for (auto device : *devices) {
        SetActiveGPU forDuration(device);
        PLD *pld = getPLD(device);
        auto rtc = device->rtc;
        pld->quadLights = 0;
        pld->dirLights = 0;
        pld->pointLights = 0;
      }
    }
    
    World::~World()
    {}

    World::PLD *World::getPLD(Device *device)
    {
      assert(device);
      assert(device->contextRank() >= 0);
      assert(device->contextRank() < perLogical.size());
      return &perLogical[device->contextRank()];
    }
    
    World::DD World::getDD(Device *device) 
    {
      PLD *pld = getPLD(device);
      DD dd;
      dd.quadLights
        = (QuadLight::DD *)pld->quadLights;
      dd.numQuadLights = pld->numQuadLights;
      dd.dirLights
        = (DirLight::DD *)pld->dirLights;
      dd.numDirLights = pld->numDirLights;
      dd.pointLights
        = (PointLight::DD *)pld->pointLights;
      dd.numPointLights = pld->numPointLights;
      dd.envMapLight
        = envMapLight.light
        ? envMapLight.light->getDD(device,envMapLight.xfm)
        : EnvMapLight::DD{};
      dd.rank = slotContext->context->myRank();
      
      dd.samplers  = slotContext->samplerRegistry->getDD(device);
      dd.materials = slotContext->materialRegistry->getDD(device);
      
      for (int i=0;i<5;i++)
        dd.instanceAttributes[i]
          = instanceAttributes[i]
          ? (const rtc::float4*)instanceAttributes[i]->getDD(device)
          : nullptr;
      return dd;
    }

    void World::set(const std::vector<QuadLight::DD> &quadLights)
    {
      for (auto device : *devices) {
        SetActiveGPU forDuration(device);
        auto pld = getPLD(device);
        auto rtc = device->rtc;
        rtc->freeMem(pld->quadLights);
        size_t numBytes = quadLights.size()*sizeof(quadLights[0]);
        pld->quadLights = (QuadLight::DD*)rtc->allocMem(numBytes);
        rtc->copy(pld->quadLights,quadLights.data(),numBytes);
        pld->numQuadLights = (int)quadLights.size();
      }
    }
    
    void World::set(const std::vector<DirLight::DD> &dirLights)
    {
      for (auto device : *devices) {
        SetActiveGPU forDuration(device);
        auto pld = getPLD(device);
        auto rtc = device->rtc;
        rtc->freeMem(pld->dirLights);
        size_t numBytes = dirLights.size()*sizeof(dirLights[0]);
        pld->dirLights = (DirLight::DD*)rtc->allocMem(numBytes);
        rtc->copy(pld->dirLights,dirLights.data(),numBytes);
        pld->numDirLights = (int)dirLights.size();
      }
    }

    void World::set(const std::vector<PointLight::DD> &pointLights)
    {
      for (auto device : *devices) {
        SetActiveGPU forDuration(device);
        auto pld = getPLD(device);
        auto rtc = device->rtc;
        rtc->freeMem(pld->pointLights);
        size_t numBytes = pointLights.size()*sizeof(pointLights[0]);
        pld->pointLights = (PointLight::DD*)rtc->allocMem(numBytes);
        rtc->copy(pld->pointLights,pointLights.data(),numBytes);
        pld->numPointLights = (int)pointLights.size();
      }
    }

    void World::set(EnvMapLight::SP envMapLight, const affine3f &xfm)
    {
      this->envMapLight.light = envMapLight;
      this->envMapLight.xfm = xfm;
    }

  } // ::BARNEY_NS::render
} // ::BARNEY_NS
