// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "native/material/Glass.h"
#include "native/material/DeviceMaterial.h"

namespace BARNEY_NS {
  namespace native {

    Glass::Glass(LDGContext *context)
      : HostMaterial(context)
    {}

    DeviceMaterial Glass::getDD(Device *device)
    {
      DeviceMaterial dd;
      dd.type = DeviceMaterial::TYPE_Glass;
      // Glass has no cut-out opacity; coverage stays fully opaque.
      packCoverage(dd, /*opacityIdenticallyOne*/true);

      dd.glass.ior              = ior.getDD(device);
      dd.glass.attenuationColor = attenuationColor.getDD(device);

      return dd;
    }

    bool Glass::setObject(const std::string &member, const Object::SP &value)
    {
      if (HostMaterial::setObject(member,value)) return true;

      Sampler::SP sampler = value ? value->as<Sampler>() : Sampler::SP();
      if (member == "ior")
        { ior.set(sampler); return true; }
      if (member == "attenuationColor")
        { attenuationColor.set(sampler); return true; }

      return false;
    }

    bool Glass::setString(const std::string &member, const std::string &value)
    {
      if (HostMaterial::setString(member,value)) return true;

      if (member == "ior")
        { ior.set(value); return true; }
      if (member == "attenuationColor")
        { attenuationColor.set(value); return true; }

      return false;
    }

    bool Glass::set1f(const std::string &member, const float &value)
    {
      if (HostMaterial::set1f(member,value)) return true;

      if (member == "ior")
        { ior.set(value); return true; }

      return false;
    }

    bool Glass::set3f(const std::string &member, const vec3f &value)
    {
      if (HostMaterial::set3f(member,value)) return true;

      if (member == "attenuationColor")
        { attenuationColor.set(value); return true; }

      return false;
    }

    bool Glass::set4f(const std::string &member, const vec4f &value)
    {
      if (HostMaterial::set4f(member,value)) return true;

      if (member == "attenuationColor")
        { attenuationColor.set(value); return true; }

      return false;
    }

  }
}
