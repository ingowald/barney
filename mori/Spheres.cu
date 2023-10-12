// ======================================================================== //
// Copyright 2023-2023 Ingo Wald                                            //
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

#include "mori/Spheres.h"

namespace mori {
  
  extern "C" char SpheresProgs_ptx[];
  
  Spheres::Spheres(DeviceContext *device,
                   const Material &material,
                   const vec3f *origins,
                   int numOrigins,
                   const float *radii,
                   float defaultRadius)
    : Geom(device,material),
      defaultRadius(defaultRadius)
  {
    OWLGeomType gt = device->getOrCreateTypeFor
      ("Spheres",Spheres::createGeomType);
    owlGeom = owlGeomCreate(device->owlContext,gt);
    originsBuffer = owlDeviceBufferCreate
      (device->owlContext,OWL_FLOAT3,numOrigins,origins);
    
    owlGeomSetRaw(owlGeom,"material",&material);
    owlGeomSet1f(owlGeom,"defaultRadius",defaultRadius);
    owlGeomSetBuffer(owlGeom,"origins",originsBuffer);
  }

  OWLGeomType Spheres::createGeomType(DeviceContext *device)
  {
    static OWLVarDecl params[]
      = {
         { "material", OWL_USER_TYPE(Material), OWL_OFFSETOF(OnDev,material) },
         { "radii", OWL_BUFPTR, OWL_OFFSETOF(OnDev,radii) },
         { "defaultRadius", OWL_FLOAT, OWL_OFFSETOF(OnDev,defaultRadius) },
         { "origins", OWL_BUFPTR, OWL_OFFSETOF(OnDev,origins) },
         { nullptr }
    };
    OWLModule module = owlModuleCreate
      (device->owlContext,SpheresProgs_ptx);
    OWLGeomType gt = owlGeomTypeCreate
      (device->owlContext,OWL_GEOM_USER,sizeof(Spheres::OnDev),
       params,-1);
    owlGeomTypeSetBoundsProg(gt,module,"SpheresBounds");
    owlGeomTypeSetIntersectProg(gt,/*ray type*/0,module,"SpheresIsec");
    owlGeomTypeSetClosestHit(gt,/*ray type*/0,module,"SpheresCH");
    
    return gt;
  }
  
}
