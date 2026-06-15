// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#include "barney/volume/Volume.h"
#include "barney/ModelSlot.h"
#include "barney/barneyConfig.h"
#if BARNEY_USE_MULTI_SCATTERING
#include "barney/common/math.h"
#else
#include "barney/volume/ScalarField.h"
#endif

namespace BARNEY_NS {

  Volume::PLD *Volume::getPLD(Device *device)
  { return &perLogical[device->contextRank()]; }

#if BARNEY_USE_MULTI_SCATTERING

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
    if (member == "principledDensity") {
      principled.density = value;
      needsMajorantRebuild = true;
      return true;
    }
    if (member == "principledDensityThreshold") {
      principled.densityThreshold = max(value, 0.f);
      needsMajorantRebuild = true;
      return true;
    }
    if (member == "principledDensityScale") {
      principled.densityScale = value > 0.f ? value : 1.f;
      needsMajorantRebuild = true;
      return true;
    }
    if (member == "principledEmissionStrength") {
      principled.emissionStrength = max(value, 0.f);
      return true;
    }
    if (member == "principledBlackbodyIntensity") {
      principled.blackbodyIntensity = max(value, 0.f);
      return true;
    }
    if (member == "principledTemperature") {
      principled.temperature = max(value, 0.f);
      return true;
    }
    return false;
  }

  bool Volume::set2f(const std::string &member,
                     const vec2f &value)
  {
    if (member == "principledValueRange") {
      principled.valueRange = range1f(value.x, value.y);
      needsMajorantRebuild = true;
      return true;
    }
    return false;
  }

  bool Volume::set3f(const std::string &member,
                     const vec3f &value)
  {
    if (member == "principledScatterColor") {
      principled.scatterColor = max(value, vec3f(0.f));
      needsMajorantRebuild = true;
      return true;
    }
    if (member == "principledAbsorptionColor") {
      principled.absorptionColor = max(value, vec3f(0.f));
      needsMajorantRebuild = true;
      return true;
    }
    if (member == "principledEmissionColor") {
      principled.emissionColor = max(value, vec3f(0.f));
      return true;
    }
    if (member == "principledBlackbodyTint") {
      principled.blackbodyTint = max(value, vec3f(0.f));
      return true;
    }
    return false;
  }

#else

  void Volume::setXF(const range1f &domain,
                     const bn_float4 *_values,
                     int numValues,
                     float baseDensity) 
  {
    std::vector<vec4f> values(numValues);
    memcpy(values.data(),_values,numValues*sizeof(*_values));
    xf.set(domain,values,baseDensity);
  }

  bool Volume::set1i(const std::string &member,
                     const int   &value) 
  {
    if (member == "userID") {
      userID = value;
      return true; 
    } 
    
    return false;
  }

#endif
  
  inline ScalarField::SP assertNotNull(const ScalarField::SP &s)
  { assert(s); return s; }
  
  inline ScalarField *assertNotNull(ScalarField *s)
  { assert(s); return s; }
  
  Volume::Volume(ScalarField::SP sf)
    : barney_api::Volume(sf->context),
      sf(sf),
      xf((Context*)sf->context,sf->devices),
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
#if BARNEY_USE_MULTI_SCATTERING
    needsMajorantRebuild = false;
#endif
    for (auto device : *devices)
      device->sbtDirty = true;
  }

}
