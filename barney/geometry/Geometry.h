// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include "barney/Object.h"
#include "barney/render/Ray.h"
#include "barney/render/HitAttributes.h"
#include "barney/render/GeometryAttributes.h"
#include "barney/material/Material.h"
#include "barney/render/OptixGlobals.h"
#include "barney/render/World.h"

namespace BARNEY_NS {
  
  struct ModelSlot;
  using render::GeometryAttribute;
  using render::GeometryAttributes;
  using render::HostMaterial;
  
  struct Geometry : public barney_api::Geometry {
    typedef std::shared_ptr<Geometry> SP;

    struct DD {
      template<typename InterpolatePerVertex>
      inline __rtc_device
      void setHitAttributes(render::HitAttributes &hit,
                            const InterpolatePerVertex &interpolate,
                            const render::World::DD &world,
                            bool dbg=false) const;

      render::GeometryAttributes::DD attributes;
      int userID;
      int materialID;
    };
    
    Geometry(Context *context,
             DevGroup::SP devices);
    virtual ~Geometry();

    static Geometry::SP create(Context *context,
                               DevGroup::SP devices,
                               const std::string &type);
    
    /*! pretty-printer for printf-debugging */
    std::string toString() const override
    { return "Geometry{}"; }

    /*! ask this geometry to build whatever owl geoms it needs to build */
    virtual void build() {}

    void setAttributesOn(Geometry::DD &dd,
                         Device *device);
    void writeDD(Geometry::DD &dd,
                Device *device);

    bool set1i(const std::string &member,
               const int   &value) override;
    bool set1f(const std::string &member,
               const float &value) override;
    bool set3f(const std::string &member,
               const vec3f &value) override;
    bool set4f(const std::string &member,
               const vec4f &value) override;
    bool setData(const std::string &member,
                 const barney_api::Data::SP &value) override;
    bool setObject(const std::string &member,
                   const Object::SP &value) override;
    
    HostMaterial::SP getMaterial() const;
    void setMaterial(HostMaterial::SP);

    struct PLD {
      std::vector<rtc::Geom *>  triangleGeoms;
      std::vector<rtc::Geom *>  userGeoms;
      // std::vector<rtc::Group *> generatedGroups;
    };
    PLD *getPLD(Device *device);
    std::vector<PLD> perLogical;
    DevGroup::SP const devices;
  protected:
    render::HostMaterial::SP material;

    render::GeometryAttributes attributes;
    int userID = 0;
  };

  template<typename InterpolatePerVertex>
  inline __rtc_device
  void Geometry::DD::setHitAttributes(render::HitAttributes &hit,
                                      const InterpolatePerVertex &interpolate,
                                      const render::World::DD &world,
                                      bool dbg) const
  {
    auto set = [&](vec4f &out,
                   const GeometryAttribute::DD &in,
                   const rtc::float4 *instanceAttribute,
                   bool dbg=false)
    {
      switch(in.scope) {
      case GeometryAttribute::INVALID:
        /* if the _geometry_ doesn't have an attribute set, it can
           still come from an instance */
        if (instanceAttribute)
          out = rtc::load(instanceAttribute[hit.instID]);
        else 
          /* nothing - leave default */
          ;
        break;
      case GeometryAttribute::CONSTANT:
        out = in.value;
        // out = rtc::load(in.value);
        break;
      case GeometryAttribute::PER_PRIM:
        out = in.fromArray.valueAt(hit.primID);
        break;
      case GeometryAttribute::PER_VERTEX:
        out = interpolate(in,/*faceVarying*/false);
        break; 
      case GeometryAttribute::FACE_VARYING:
        out = interpolate(in,/*faceVarying*/true);
        break; 
      }
    };
    
    for (int i=0;i<attributes.count;i++) {
      vec4f     &out = hit.attribute[i];
      const auto &in  = this->attributes.attribute[i];
      set(out,in,world.instanceAttributes[i]);
    }
    set(hit.color,this->attributes.colorAttribute,world.instanceAttributes[4],dbg);
    set(hit.objectNormal,this->attributes.normalAttribute,nullptr,dbg);
  }
  
}
