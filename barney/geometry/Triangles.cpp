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
#include "barney/ModelSlot.h"

namespace barney {

  extern "C" char Triangles_ptx[];

  Triangles::Triangles(ModelSlot *owner)
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
      { "vertices", OWL_BUFPTR, OWL_OFFSETOF(DD,vertices) },
      { "indices", OWL_BUFPTR, OWL_OFFSETOF(DD,indices) },
      { "texcoords", OWL_BUFPTR, OWL_OFFSETOF(DD,texcoords) },
      { "normals", OWL_BUFPTR, OWL_OFFSETOF(DD,normals) },
      { "primitive.attribute0", OWL_BUFPTR, OWL_OFFSETOF(DD,primitiveAttribute[0]) },
      { "primitive.attribute1", OWL_BUFPTR, OWL_OFFSETOF(DD,primitiveAttribute[1]) },
      { "primitive.attribute2", OWL_BUFPTR, OWL_OFFSETOF(DD,primitiveAttribute[2]) },
      { "primitive.attribute3", OWL_BUFPTR, OWL_OFFSETOF(DD,primitiveAttribute[3]) },
      { "primitive.attribute4", OWL_BUFPTR, OWL_OFFSETOF(DD,primitiveAttribute[4]) },
      { "vertex.attribute0", OWL_BUFPTR, OWL_OFFSETOF(DD,vertexAttribute[0]) },
      { "vertex.attribute1", OWL_BUFPTR, OWL_OFFSETOF(DD,vertexAttribute[1]) },
      { "vertex.attribute2", OWL_BUFPTR, OWL_OFFSETOF(DD,vertexAttribute[2]) },
      { "vertex.attribute3", OWL_BUFPTR, OWL_OFFSETOF(DD,vertexAttribute[3]) },
      { "vertex.attribute4", OWL_BUFPTR, OWL_OFFSETOF(DD,vertexAttribute[4]) },
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
    if (member == "vertex.attribute0") {
      vertexAttribute0 = value->as<PODData>();
      return true;
    }
    if (member == "vertex.attribute1") {
      vertexAttribute1 = value->as<PODData>();
      return true;
    }
    if (member == "vertex.attribute2") {
      vertexAttribute2 = value->as<PODData>();
      return true;
    }
    if (member == "vertex.attribute3") {
      vertexAttribute3 = value->as<PODData>();
      return true;
    }
    if (member == "vertex.attribute4") {
      vertexAttribute4 = value->as<PODData>();
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

    OWLGeom   geom = triangleGeoms[0];
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
    // PING; PRINT(normals);
<<<<<<< HEAD
=======

    OWLBuffer primitiveAttribute0Buffer
      = attribute0
      ? attribute0->owl
      : 0;
    OWLBuffer primitiveAttribute1Buffer
      = attribute1
      ? attribute1->owl
      : 0;
    OWLBuffer primitiveAttribute2Buffer
      = attribute2
      ? attribute2->owl
      : 0;
    OWLBuffer primitiveAttribute3Buffer
      = attribute3
      ? attribute3->owl
      : 0;
    OWLBuffer primitiveAttribute4Buffer
      = attribute4
      ? attribute4->owl
      : 0;

    OWLBuffer vertexAttribute0Buffer
      = vertexAttribute0
      ? vertexAttribute0->owl
      : 0;
    OWLBuffer vertexAttribute1Buffer
      = vertexAttribute1
      ? vertexAttribute1->owl
      : 0;
    OWLBuffer vertexAttribute2Buffer
      = vertexAttribute2
      ? vertexAttribute2->owl
      : 0;
    OWLBuffer vertexAttribute3Buffer
      = vertexAttribute3
      ? vertexAttribute3->owl
      : 0;
    OWLBuffer vertexAttribute4Buffer
      = vertexAttribute4
      ? vertexAttribute4->owl
      : 0;
>>>>>>> iw/temp-merge-pt

    int numVertices = vertices->count;
    int numIndices  = indices->count;
    owlTrianglesSetVertices(geom,verticesBuffer,
                            numVertices,sizeof(float3),0);
    owlTrianglesSetIndices(geom,indicesBuffer,
                           numIndices,sizeof(int3),0);
    owlGeomSetBuffer(geom,"vertices",verticesBuffer);
    owlGeomSetBuffer(geom,"indices",indicesBuffer);
    owlGeomSetBuffer(geom,"normals",normalsBuffer);
    owlGeomSetBuffer(geom,"texcoords",texcoordsBuffer);
    owlGeomSetBuffer(geom,"primitive.attribute0",primitiveAttribute0Buffer);
    owlGeomSetBuffer(geom,"primitive.attribute1",primitiveAttribute1Buffer);
    owlGeomSetBuffer(geom,"primitive.attribute2",primitiveAttribute2Buffer);
    owlGeomSetBuffer(geom,"primitive.attribute3",primitiveAttribute3Buffer);
    owlGeomSetBuffer(geom,"primitive.attribute4",primitiveAttribute4Buffer);
    owlGeomSetBuffer(geom,"vertex.attribute0",vertexAttribute0Buffer);
    owlGeomSetBuffer(geom,"vertex.attribute1",vertexAttribute1Buffer);
    owlGeomSetBuffer(geom,"vertex.attribute2",vertexAttribute2Buffer);
    owlGeomSetBuffer(geom,"vertex.attribute3",vertexAttribute3Buffer);
    owlGeomSetBuffer(geom,"vertex.attribute4",vertexAttribute4Buffer);
    
    material->setDeviceDataOn(geom);
  }
  
}

