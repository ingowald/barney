// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "native/material/NVisii.h"
#include "native/material/DeviceMaterial.h"

namespace BARNEY_NS {
  namespace native {

    NVisii::NVisii(LDGContext *context)
      : HostMaterial(context)
    {}

    DeviceMaterial NVisii::getDD(Device *device)
    {
      DeviceMaterial dd;
      dd.type = DeviceMaterial::TYPE_NVisii;

      dd.nvisii.baseColor            = baseColor.getDD(device);
      dd.nvisii.subsurfaceColor      = subsurfaceColor.getDD(device);
      dd.nvisii.metallic             = metallic.getDD(device);
      dd.nvisii.specular             = specular.getDD(device);
      dd.nvisii.roughness            = roughness.getDD(device);
      dd.nvisii.specularTint         = specularTint.getDD(device);
      dd.nvisii.anisotropy           = anisotropy.getDD(device);
      dd.nvisii.sheen                = sheen.getDD(device);
      dd.nvisii.sheenTint            = sheenTint.getDD(device);
      dd.nvisii.clearcoat            = clearcoat.getDD(device);
      dd.nvisii.clearcoatGloss       = clearcoatGloss.getDD(device);
      dd.nvisii.ior                  = ior.getDD(device);
      dd.nvisii.specularTransmission = specularTransmission.getDD(device);
      dd.nvisii.transmissionRoughness= transmissionRoughness.getDD(device);
      dd.nvisii.flatness             = flatness.getDD(device);
      dd.nvisii.opacity              = opacity.getDD(device);

      return dd;
    }

    bool NVisii::setObject(const std::string &member, const Object::SP &value)
    {
      if (HostMaterial::setObject(member,value)) return true;

      Sampler::SP sampler = value ? value->as<Sampler>() : Sampler::SP();
      if (member == "baseColor")
        { baseColor.set(sampler); return true; }
      if (member == "subsurfaceColor")
        { subsurfaceColor.set(sampler); return true; }
      if (member == "metallic")
        { metallic.set(sampler); return true; }
      if (member == "specular")
        { specular.set(sampler); return true; }
      if (member == "roughness")
        { roughness.set(sampler); return true; }
      if (member == "specularTint")
        { specularTint.set(sampler); return true; }
      if (member == "anisotropy")
        { anisotropy.set(sampler); return true; }
      if (member == "sheen")
        { sheen.set(sampler); return true; }
      if (member == "sheenTint")
        { sheenTint.set(sampler); return true; }
      if (member == "clearcoat")
        { clearcoat.set(sampler); return true; }
      if (member == "clearcoatGloss")
        { clearcoatGloss.set(sampler); return true; }
      if (member == "ior")
        { ior.set(sampler); return true; }
      if (member == "specularTransmission")
        { specularTransmission.set(sampler); return true; }
      if (member == "transmissionRoughness")
        { transmissionRoughness.set(sampler); return true; }
      if (member == "flatness")
        { flatness.set(sampler); return true; }
      if (member == "opacity")
        { opacity.set(sampler); return true; }

      return false;
    }

    bool NVisii::setString(const std::string &member, const std::string &value)
    {
      if (HostMaterial::setString(member,value)) return true;

      if (member == "baseColor")
        { baseColor.set(value); return true; }
      if (member == "subsurfaceColor")
        { subsurfaceColor.set(value); return true; }
      if (member == "metallic")
        { metallic.set(value); return true; }
      if (member == "specular")
        { specular.set(value); return true; }
      if (member == "roughness")
        { roughness.set(value); return true; }
      if (member == "specularTint")
        { specularTint.set(value); return true; }
      if (member == "anisotropy")
        { anisotropy.set(value); return true; }
      if (member == "sheen")
        { sheen.set(value); return true; }
      if (member == "sheenTint")
        { sheenTint.set(value); return true; }
      if (member == "clearcoat")
        { clearcoat.set(value); return true; }
      if (member == "clearcoatGloss")
        { clearcoatGloss.set(value); return true; }
      if (member == "ior")
        { ior.set(value); return true; }
      if (member == "specularTransmission")
        { specularTransmission.set(value); return true; }
      if (member == "transmissionRoughness")
        { transmissionRoughness.set(value); return true; }
      if (member == "flatness")
        { flatness.set(value); return true; }
      if (member == "opacity")
        { opacity.set(value); return true; }

      return false;
    }

    bool NVisii::set1f(const std::string &member, const float &value)
    {
      if (HostMaterial::set1f(member,value)) return true;

      if (member == "metallic")
        { metallic.set(value); return true; }
      if (member == "specular")
        { specular.set(value); return true; }
      if (member == "roughness")
        { roughness.set(value); return true; }
      if (member == "specularTint")
        { specularTint.set(value); return true; }
      if (member == "anisotropy")
        { anisotropy.set(value); return true; }
      if (member == "sheen")
        { sheen.set(value); return true; }
      if (member == "sheenTint")
        { sheenTint.set(value); return true; }
      if (member == "clearcoat")
        { clearcoat.set(value); return true; }
      if (member == "clearcoatGloss")
        { clearcoatGloss.set(value); return true; }
      if (member == "ior")
        { ior.set(value); return true; }
      if (member == "specularTransmission")
        { specularTransmission.set(value); return true; }
      if (member == "transmissionRoughness")
        { transmissionRoughness.set(value); return true; }
      if (member == "flatness")
        { flatness.set(value); return true; }
      if (member == "opacity")
        { opacity.set(value); return true; }

      return false;
    }

    bool NVisii::set3f(const std::string &member, const vec3f &value)
    {
      if (HostMaterial::set3f(member,value)) return true;

      if (member == "baseColor")
        { baseColor.set(value); return true; }
      if (member == "subsurfaceColor")
        { subsurfaceColor.set(value); return true; }

      return false;
    }

    bool NVisii::set4f(const std::string &member, const vec4f &value)
    {
      if (HostMaterial::set4f(member,value)) return true;

      if (member == "baseColor")
        { baseColor.set(value); return true; }
      if (member == "subsurfaceColor")
        { subsurfaceColor.set(value); return true; }

      return false;
    }

  }
}
