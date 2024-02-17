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

#include "barney/geometry/Triangles.h"
#include "barney/DataGroup.h"

namespace barney {

  extern "C" char Triangles_ptx[];

  Triangles::Triangles(DataGroup *owner)
    : Geometry(owner)
  {}
  

  Triangles::~Triangles()
  {}
  
  OWLGeomType Triangles::createGeomType(DevGroup *devGroup)
  {
    std::cout << OWL_TERMINAL_GREEN
              << "creating 'Triangles' geometry type"
              << OWL_TERMINAL_DEFAULT << std::endl;

    std::vector<OWLVarDecl> params
      = {
      // { "material", OWL_USER_TYPE(Material), OWL_OFFSETOF(DD,material) },
      { "vertices", OWL_BUFPTR, OWL_OFFSETOF(DD,vertices) },
      { "indices", OWL_BUFPTR, OWL_OFFSETOF(DD,indices) },
      { "texcoords", OWL_BUFPTR, OWL_OFFSETOF(DD,texcoords) },
      { "normals", OWL_BUFPTR, OWL_OFFSETOF(DD,normals) },
    };
    Geometry::addVars(params,0);
    
    OWLModule module = owlModuleCreate
      (devGroup->owl,Triangles_ptx);
    OWLGeomType gt = owlGeomTypeCreate
      (devGroup->owl,OWL_GEOM_TRIANGLES,sizeof(Triangles::DD),
       params.data(),(int)params.size());
    owlGeomTypeSetClosestHit(gt,/*ray type*/0,module,"TrianglesCH");
    owlGeomTypeSetAnyHit(gt,/*ray type*/0,module,"TrianglesAH");
    owlBuildPrograms(devGroup->owl);
    owlModuleRelease(module);
    return gt;
  }


  bool Triangles::setData(const std::string &member, const Data::SP &value)
  {
    if (Geometry::setData(member,value))
      return true;
    if (member == "vertices") {
      vertices = value->as<PODData>();
      return true;
    }
    if (member == "indices") {
      indices = value->as<PODData>();
      return true;
    }
    if (member == "normals") {
      normals = value->as<PODData>();
      return true;
    }
    if (member == "texcoords") {
      texcoords = value->as<PODData>();
      return true;
    }
    return false;
  }
  
  void Triangles::commit() 
  {
    if (triangleGeoms.empty()) {
      OWLGeomType gt = owner->devGroup->getOrCreateGeomTypeFor
        ("Triangles",Triangles::createGeomType);
      OWLGeom geom = owlGeomCreate(owner->devGroup->owl,gt);
      triangleGeoms = { geom };
    }

    OWLGeom geom = triangleGeoms[0];
    OWLBuffer verticesBuffer = vertices->owl;
    OWLBuffer indicesBuffer = indices->owl;
    OWLBuffer texcoordsBuffer
      = texcoords
      ? texcoords->owl
      : 0;
    OWLBuffer normalsBuffer
      = normals
      ? normals->owl
      : 0;

    int numVertices = vertices->count;
    int numIndices  = indices->count;
    PRINT(numVertices);
    PRINT(numIndices);
    owlTrianglesSetVertices(geom,verticesBuffer,
                            numVertices,sizeof(float3),0);
    owlTrianglesSetIndices(geom,indicesBuffer,
                           numIndices,sizeof(int3),0);
    // Geometry::setMaterial(geom);
    // owlGeomSetRaw(geom,"material",&material);
    owlGeomSetBuffer(geom,"vertices",verticesBuffer);
    owlGeomSetBuffer(geom,"indices",indicesBuffer);
    owlGeomSetBuffer(geom,"normals",normalsBuffer);
    owlGeomSetBuffer(geom,"texcoords",texcoordsBuffer);
    
    material->set(geom);
  }

  
}

