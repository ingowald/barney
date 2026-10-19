// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "native/volume/Volume.h"
#include "native/ModelSlot.h"
#include "native/volume/ScalarField.h"

namespace BARNEY_NS {
  namespace native {

    Volume::PLD *Volume::getPLD(Device *device)
    { return &perLogical[device->contextRank()]; }

    void Volume::setXF(const range1f &domain,
                       const bn_float4 *_values,
                       int numValues,
                       float baseDensity) 
    {
      std::vector<vec4f> values(numValues);
      memcpy(values.data(),_values,numValues*sizeof(*_values));
      xf.set(domain,values,baseDensity);
      needsMajorantRebuild = true;
    }

    void Volume::setEmissionXF(const range1f &domain,
                               const bn_float4 *_values,
                               int numValues)
    {
      if (numValues <= 0) {
        clearEmissionXF();
        return;
      }
      std::vector<vec4f> values(numValues);
      memcpy(values.data(),_values,numValues*sizeof(*_values));
      if (values.size() == 1)
        values.push_back(values.front());
      emissionXf.set(domain,values,1.f);
      emissionXfActive = true;
    }

    void Volume::clearEmissionXF()
    {
      emissionXfActive = false;
    }

    void Volume::commit()
    {
      if (!accel)
        return;
      if (needsMajorantRebuild && sf->mcGrid && sf->mcGrid->built()) {
        accel->rebuildMajorantsOnly();
        needsMajorantRebuild = false;
      }
      accel->refreshDeviceData();
    }

    bool Volume::set1i(const std::string &member,
                       const int   &value) 
    {
      if (member == "userID") {
        userID = value;
        return true; 
      }
      if (member == "principledVolume") {
        principled.enabled = value ? 1 : 0;
        needsMajorantRebuild = true;
        return true;
      }
    
      return false;
    }

    bool Volume::set1f(const std::string &member,
                       const float &value) 
    {
      if (member == "anisotropy") {
        anisotropy = clamp(value, -0.99f, 0.99f);
        return true;
      }
      if (member == "scatteringAlbedo") {
        scatteringAlbedo = clamp(value, 0.f, 1.f);
        return true;
      }
      if (member == "principledEmissionStrength") {
        principled.emissionStrength = max(value, 0.f);
        return true;
      }
      return false;
    }

    bool Volume::set2f(const std::string &member,
                       const vec2f &value)
    {
      (void)member;
      (void)value;
      return false;
    }

    bool Volume::set3f(const std::string &member,
                       const vec3f &value)
    {
      (void)member;
      (void)value;
      return false;
    }

    inline ScalarField::SP assertNotNull(const ScalarField::SP &s)
    { assert(s); return s; }
  
    inline ScalarField *assertNotNull(ScalarField *s)
    { assert(s); return s; }
  
    Volume::Volume(ScalarField::SP sf)
      : Object(sf->context),
        sf(sf),
        xf((Context*)sf->context,sf->devices),
        emissionXf((Context*)sf->context,sf->devices),
        devices(sf->devices)
    {
      accel = sf->createAccel(this);
      perLogical.resize(devices->numLogical);
    }

    const TransferFunction *VolumeAccel::getXF() const { return &volume->xf; }
  
    void Volume::build(bool full_rebuild)
    {
      assert(accel);
      accel->build(full_rebuild);
      needsMajorantRebuild = false;
      for (auto device : *devices)
        device->sbtDirty = true;
    }

  }
}
