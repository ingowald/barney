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

  Cylinders::Cylinders(DataGroup *owner)
  : Geometry(owner)
  {
// ,
//                        const vec3f *vertices,
//                        int          numVertices,
//                        const vec3f *colors,
//                        bool         colorPerVertex,
//                        const vec2i *indices,
//                        int          numIndices,
//                        const float *radii,
//                        bool         radiusPerVertex,
//                        float        defaultRadius    
//     verticesBuffer = owlDeviceBufferCreate
//       (owner->devGroup->owl,
//        OWL_FLOAT3,numVertices,vertices);
//     std::vector<vec2i> explicitIndices;
//     if (!indices) {
//       for (int i=0;i<numVertices/2;i++)
//         explicitIndices.push_back(2*i+vec2i(0,1));
//       indices = explicitIndices.data();
//       numIndices = (int)explicitIndices.size();
//     }
//     std::vector<float> explicitRadii;
//     if (!radii) {
//       for (int i=0;i<numIndices;i++)
//         explicitRadii.push_back(defaultRadius);
//       radii = explicitRadii.data();
//     }
//     indicesBuffer = owlDeviceBufferCreate
//       (owner->devGroup->owl,
//        OWL_INT2,numIndices,indices);
//     verticesBuffer = owlDeviceBufferCreate
//       (owner->devGroup->owl,
//        OWL_FLOAT3,numVertices,vertices);
//     if (colors)
//       colorsBuffer = owlDeviceBufferCreate
//         (owner->devGroup->owl,
//          OWL_FLOAT3,colorPerVertex?numVertices:numIndices,
//          colors);
//     radiiBuffer = owlDeviceBufferCreate
//       (owner->devGroup->owl,
//        OWL_FLOAT,numIndices,radii);

//     OWLGeomType gt = owner->devGroup->getOrCreateGeomTypeFor
//       ("Cylinders",Cylinders::createGeomType);
//     OWLGeom geom = owlGeomCreate(owner->devGroup->owl,gt);
    
//     owlGeomSetPrimCount(geom,numIndices);
//     owlGeomSetBuffer(geom,"colors",colorsBuffer);
//     owlGeomSetBuffer(geom,"vertices",verticesBuffer);
//     owlGeomSetBuffer(geom,"indices",indicesBuffer);
//     owlGeomSetBuffer(geom,"radii",radiiBuffer);
//     owlGeomSet1i(geom,"colorPerVertex",(int)colorPerVertex);
//     owlGeomSet1i(geom,"radiusPerVertex",(int)radiusPerVertex);
    
//     userGeoms.push_back(geom);
  }

  OWLGeomType Cylinders::createGeomType(DevGroup *devGroup)
  {
    std::cout << OWL_TERMINAL_GREEN
              << "creating 'Cylinders' geometry type"
              << OWL_TERMINAL_DEFAULT << std::endl;
    
    std::vector<OWLVarDecl> params
      = {
         { "radii", OWL_BUFPTR, OWL_OFFSETOF(DD,radii) },
         { "vertices", OWL_BUFPTR, OWL_OFFSETOF(DD,vertices) },
         { "colors", OWL_BUFPTR, OWL_OFFSETOF(DD,colors) },
         { "indices", OWL_BUFPTR, OWL_OFFSETOF(DD,indices) },
         { "colorPerVertex", OWL_INT, OWL_OFFSETOF(DD,colorPerVertex) },
         { "radiusPerVertex", OWL_INT, OWL_OFFSETOF(DD,radiusPerVertex) },
    };
    Geometry::addVars(params,0);
    OWLModule module = owlModuleCreate
      (devGroup->owl,Cylinders_ptx);
    OWLGeomType gt = owlGeomTypeCreate
      (devGroup->owl,OWL_GEOM_USER,sizeof(Cylinders::DD),
       params.data(), (int)params.size());
    owlGeomTypeSetBoundsProg(gt,module,"CylindersBounds");
    owlGeomTypeSetIntersectProg(gt,/*ray type*/0,module,"CylindersIsec");
    owlGeomTypeSetClosestHit(gt,/*ray type*/0,module,"CylindersCH");
    owlBuildPrograms(devGroup->owl);
    
    return gt;
  }
  
  void Cylinders::commit()
  {
    if (userGeoms.empty()) {
      OWLGeomType gt = owner->devGroup->getOrCreateGeomTypeFor
        ("Cylinders",Cylinders::createGeomType);
      OWLGeom geom = owlGeomCreate(owner->devGroup->owl,gt);
      userGeoms.push_back(geom);
    }
    OWLGeom geom = userGeoms[0];
    
    Geometry::commit();
    owlGeomSet1i(geom,"colorPerVertex",colorPerVertex);
    owlGeomSet1i(geom,"radiusPerVertex",radiusPerVertex);
    owlGeomSetBuffer(geom,"vertices",vertices?vertices->owl:0);
    owlGeomSetBuffer(geom,"indices",indices?indices->owl:0);
    owlGeomSetBuffer(geom,"colors",colors?colors->owl:0);
    owlGeomSetBuffer(geom,"radii",radii?radii->owl:0);
    int numIndices = indices->count;
    PRINT(numIndices);
    owlGeomSetPrimCount(geom,numIndices);
    material->set(geom);
  } 

  bool Cylinders::set1i(const std::string &member, const int &value)
  {
    if (Geometry::set1i(member,value))
      return true;
    if (member == "radiusPerVertex") {
      radiusPerVertex = value;
      return true;
    }
    if (member == "colorPerVertex") {
      colorPerVertex = value;
      return true;
    }
    return false;
  }
  
  bool Cylinders::set1f(const std::string &member, const float &value)
  {
    if (Geometry::set1f(member,value))
      return true;
    return false;
  }
  
  bool Cylinders::setData(const std::string &member, const Data::SP &value)
  {
    if (Geometry::setData(member,value))
      return true;
    if (member == "colors") {
      colors = value->as<PODData>();
      return true;
    }
    if (member == "vertices") {
      vertices = value->as<PODData>();
      PRINT(vertices->count);
      return true;
    }
    if (member == "indices") {
      indices = value->as<PODData>();
      PRINT(indices);
      PRINT(indices->count);
      return true;
    }
    if (member == "radii") {
      radii = value->as<PODData>();
      PRINT(radii->count);
      return true;
    }
    return false;
  }

  bool Cylinders::setObject(const std::string &member, const Object::SP &value)
  {
    if (Geometry::setObject(member,value))
      return true;
    return false;
  }

}

