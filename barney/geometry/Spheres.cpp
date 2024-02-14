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

#define BUFFER_CREATE owlDeviceBufferCreate
// #define BUFFER_CREATE owlManagedMemoryBufferCreate

namespace barney {
  
  extern "C" char Spheres_ptx[];
  
  OWLGeomType Spheres::createGeomType(DevGroup *devGroup)
  {
    std::cout << OWL_TERMINAL_GREEN
              << "creating 'Spheres' geometry type"
              << OWL_TERMINAL_DEFAULT << std::endl;
    
    std::vector<OWLVarDecl> params
      = {
         { "radii", OWL_BUFPTR, OWL_OFFSETOF(DD,radii) },
         { "defaultRadius", OWL_FLOAT, OWL_OFFSETOF(DD,defaultRadius) },
         { "origins", OWL_BUFPTR, OWL_OFFSETOF(DD,origins) },
         { "colors", OWL_BUFPTR, OWL_OFFSETOF(DD,colors) },
    };
    Geometry::addVars(params,0);
    OWLModule module = owlModuleCreate
      (devGroup->owl,Spheres_ptx);
    OWLGeomType gt = owlGeomTypeCreate
      (devGroup->owl,OWL_GEOM_USER,sizeof(Spheres::DD),
       params.data(),(int)params.size());
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
                   const vec3f *colors,
                   const float *radii,
                   float defaultRadius)
    : Geometry(owner,material)
  {
    OWLGeomType gt = owner->devGroup->getOrCreateGeomTypeFor
      ("Spheres",Spheres::createGeomType);
    OWLGeom geom = owlGeomCreate(owner->devGroup->owl,gt);
    originsBuffer = BUFFER_CREATE
      (owner->devGroup->owl,OWL_FLOAT3,numOrigins,origins);
    if (colors)
      colorsBuffer = BUFFER_CREATE
        (owner->devGroup->owl,OWL_FLOAT3,numOrigins,colors);

    Geometry::setMaterial(geom);
    owlGeomSet1f(geom,"defaultRadius",defaultRadius);
    owlGeomSetBuffer(geom,"origins",originsBuffer);
    owlGeomSetBuffer(geom,"colors",colorsBuffer);
    owlGeomSetPrimCount(geom,numOrigins);
    
    userGeoms.push_back(geom);
  }

}

