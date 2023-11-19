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

#include "barney/geometry/Spheres.h"
#include "barney/Model.h"

namespace barney {
  
  extern "C" char Spheres_ptx[];
  
  OWLGeomType Spheres::createGeomType(DevGroup *devGroup)
  {
    std::cout << OWL_TERMINAL_GREEN
              << "creating 'Spheres' geometry type"
              << OWL_TERMINAL_DEFAULT << std::endl;
    
    static OWLVarDecl params[]
      = {
         { "material", OWL_USER_TYPE(Material), OWL_OFFSETOF(DD,material) },
         { "radii", OWL_BUFPTR, OWL_OFFSETOF(DD,radii) },
         { "defaultRadius", OWL_FLOAT, OWL_OFFSETOF(DD,defaultRadius) },
         { "origins", OWL_BUFPTR, OWL_OFFSETOF(DD,origins) },
         { nullptr }
    };
    OWLModule module = owlModuleCreate
      (devGroup->owl,Spheres_ptx);
    OWLGeomType gt = owlGeomTypeCreate
      (devGroup->owl,OWL_GEOM_USER,sizeof(Spheres::DD),
       params,-1);
    owlGeomTypeSetBoundsProg(gt,module,"SpheresBounds");
    owlGeomTypeSetIntersectProg(gt,/*ray type*/0,module,"SpheresIsec");
    owlGeomTypeSetClosestHit(gt,/*ray type*/0,module,"SpheresCH");
    owlBuildPrograms(devGroup->owl);
    
    return gt;
  }
  
  Spheres::Spheres(DataGroup *owner,
                   const Material &material,
                   const vec3f *origins,
                   int numOrigins,
                   const float *radii,
                   float defaultRadius)
    : Geometry(owner,material)
  {
    OWLGeomType gt = owner->devGroup->getOrCreateGeomTypeFor
      ("Spheres",Spheres::createGeomType);
    OWLGeom geom = owlGeomCreate(owner->devGroup->owl,gt);
    originsBuffer = owlDeviceBufferCreate
      (owner->devGroup->owl,OWL_FLOAT3,numOrigins,origins);
    
    owlGeomSetRaw(geom,"material",&material);
    owlGeomSet1f(geom,"defaultRadius",defaultRadius);
    owlGeomSetBuffer(geom,"origins",originsBuffer);
    owlGeomSetPrimCount(geom,numOrigins);
    
    userGeoms.push_back(geom);
  }

}

