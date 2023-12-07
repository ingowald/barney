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

#include "barney/geometry/Cylinders.h"
#include "barney/DataGroup.h"

namespace barney {

  extern "C" char Cylinders_ptx[];

  Cylinders::Cylinders(DataGroup *owner,
                       const Material &material,
                       const vec3f *points,
                       int          numPoints,
                       const vec2i *indices,
                       int          numIndices,
                       const float *radii,
                       float        defaultRadius)
  : Geometry(owner,material)
  {
    pointsBuffer = owlDeviceBufferCreate
      (owner->devGroup->owl,
       OWL_FLOAT3,numPoints,points);
    std::vector<vec2i> explicitIndices;
    if (!indices) {
      for (int i=0;i<numPoints/2;i++)
        explicitIndices.push_back(2*i+vec2i(0,1));
      indices = explicitIndices.data();
      numIndices = explicitIndices.size();
    }
    std::vector<float> explicitRadii;
    if (!radii) {
      for (int i=0;i<numIndices;i++)
        explicitRadii.push_back(defaultRadius);
      radii = explicitRadii.data();
    }
    indicesBuffer = owlDeviceBufferCreate
      (owner->devGroup->owl,
       OWL_INT2,numIndices,indices);
    pointsBuffer = owlDeviceBufferCreate
      (owner->devGroup->owl,
       OWL_FLOAT3,numPoints,points);
    radiiBuffer = owlDeviceBufferCreate
      (owner->devGroup->owl,
       OWL_FLOAT,numIndices,radii);

    OWLGeomType gt = owner->devGroup->getOrCreateGeomTypeFor
      ("Cylinders",Cylinders::createGeomType);
    OWLGeom geom = owlGeomCreate(owner->devGroup->owl,gt);
    
    owlGeomSetPrimCount(geom,numIndices);
    owlGeomSetRaw(geom,"material",&material);
    owlGeomSetBuffer(geom,"points",pointsBuffer);
    owlGeomSetBuffer(geom,"indices",indicesBuffer);
    owlGeomSetBuffer(geom,"radii",radiiBuffer);
    
    userGeoms.push_back(geom);
  }

  OWLGeomType Cylinders::createGeomType(DevGroup *devGroup)
  {
    std::cout << OWL_TERMINAL_GREEN
              << "creating 'Cylinders' geometry type"
              << OWL_TERMINAL_DEFAULT << std::endl;
    
    static OWLVarDecl params[]
      = {
         { "material", OWL_USER_TYPE(Material), OWL_OFFSETOF(DD,material) },
         { "radii", OWL_BUFPTR, OWL_OFFSETOF(DD,radii) },
         { "points", OWL_BUFPTR, OWL_OFFSETOF(DD,points) },
         { "indices", OWL_BUFPTR, OWL_OFFSETOF(DD,indices) },
         { nullptr }
    };
    OWLModule module = owlModuleCreate
      (devGroup->owl,Cylinders_ptx);
    OWLGeomType gt = owlGeomTypeCreate
      (devGroup->owl,OWL_GEOM_USER,sizeof(Cylinders::DD),
       params,-1);
    owlGeomTypeSetBoundsProg(gt,module,"CylindersBounds");
    owlGeomTypeSetIntersectProg(gt,/*ray type*/0,module,"CylindersIsec");
    owlGeomTypeSetClosestHit(gt,/*ray type*/0,module,"CylindersCH");
    owlBuildPrograms(devGroup->owl);
    
    
    return gt;
  }
  
}

