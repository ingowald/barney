// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#include "barney/geometry/IsoSurface.h"
#include "barney/ModelSlot.h"
#include "barney/Context.h"

namespace BARNEY_NS {

  IsoSurfaceAccel::IsoSurfaceAccel(IsoSurface *isoSurface)
    : isoSurface(isoSurface),
      devices(isoSurface->devices)
  {}
  
  IsoSurface::IsoSurface(Context *context, DevGroup::SP devices)
    : Geometry(context,devices)
  {}

  /*! (re-)build the accel structure for this volume, probably after
    changes to transfer functoin (or later, scalar field) */
  void IsoSurface::build()
  {
    if (!accel) 
      return;
    accel->build();
  }
  
  void IsoSurface::commit()
  {
    if (!sf)
      /* we don't have a scalar field attached - this is user error
         but let's just irgnoe and skip */
      return;

    if (!accel) {
      accel = sf->createIsoAccel(this);
      if (!accel)
        throw std::runtime_error
          ("'isoSurface' geometry support not implemented for scalar "
           "field of type '"+sf->toString()+"'");
    }

    accel->build();
  } 

  bool IsoSurface::set1f(const std::string &member, const float &value)
  {
    if (Geometry::set1f(member,value))
      return true;
    if (member == "isoValue") {
      isoValue = value;
      return true;
    }
    return false;
  }
  
  bool IsoSurface::setData(const std::string &member, const Data::SP &value)
  {
    if (Geometry::setData(member,value))
      return true;
    
    if (member == "isoValues") {
      isoValues = value ? value->as<PODData>() : PODData::SP();
      return true;
    }
    return false;
  }

  bool IsoSurface::setObject(const std::string &member, const Object::SP &value)
  {
    if (Geometry::setObject(member,value))
      return true;
    if (member == "scalarField") {
      sf = value->as<ScalarField>();
      if (!sf)
        throw std::runtime_error
          ("'scalarField' value set on 'isoSurface' object "
           "is not a scalar field");
      return true;
    }
    return false;
  }

}

