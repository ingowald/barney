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

#include "barney/geometry/IsoSurface.h"
#include "barney/ModelSlot.h"
#include "barney/Context.h"

namespace BARNEY_NS {

  // RTC_IMPORT_USER_GEOM(IsoSurface,IsoSurface,IsoSurface::DD,false,true);

  IsoSurfaceAccel::IsoSurfaceAccel(IsoSurface *isoSurface)
    : isoSurface(isoSurface),
      devices(isoSurface->devices)
  {}
  
  IsoSurface::IsoSurface(Context *context, DevGroup::SP devices)
    : Geometry(context,devices)
  {}

  /*  ! (re-)build the accel structure for this volume, probably after
      changes to transfer functoin (or later, scalar field) */
  void IsoSurface::build()
  {
    PING;
    if (!accel) {
      PING;
      return;
    }
    assert(accel);
    accel->build();
    // for (auto device : *devices)  {
    //   PLD *thisPLD = getPLD(device);
    //   IsoSurfaceAccel::PLD *accelPLD = accel->getPLD(device);
    //   this->userGroups = accelPLD->
    // }
      // device->sbtDirty = true;
  }
  
  void IsoSurface::commit()
  {
    if (!sf) return;

    if (!accel)
      accel = sf->createIsoAccel(this);

    accel->build();
    // for (auto device : *devices) {
    //   PLD *pld = getPLD(device);
    //   // pld->userGeoms  =
    //   //   pld->group

    //   // if (pld->userGeoms.empty()) {
    //   //   rtc::GeomType *gt
    //   //     = device->geomTypes.get(createGeomType_IsoSurface);
    //   //   rtc::Geom *geom = gt->createGeom();
    //   //   geom->setPrimCount(1);
    //   //   pld->userGeoms.push_back(geom);
    //   // }
    //   // rtc::Geom *geom = pld->userGeoms[0];
      
    //   // IsoSurface::DD dd;
    //   // Geometry::writeDD(dd,device);
    //   // if (isoValues) {
    //   //   dd.isoValues = (float*)isoValues->getDD(device);
    //   //   dd.numIsoValues = isoValues->count;
    //   // } else {
    //   //   dd.isoValues = nullptr;
    //   //   dd.numIsoValues = 0;
    //   // }
    //   // dd.isoValue = isoValue;
    //   // // done:
    //   // geom->setDD(&dd);
    // }
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
      PING; PRINT(sf);
      return true;
    }
    return false;
  }

}

