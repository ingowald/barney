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

  Triangles::Triangles(Context *context, int slot)
    : Geometry(context,slot)
  {}
  
  Triangles::~Triangles()
  {}
  
  rtc::GeomType *Triangles::createGeomType(DevGroup *devGroup)
  {
    if (DevGroup::logging())
    std::cout << OWL_TERMINAL_GREEN
              << "creating 'Triangles' geometry type"
              << OWL_TERMINAL_DEFAULT << std::endl;

    // std::vector<OWLVarDecl> params
    //   = {
    //   { "vertices", OWL_BUFPTR, OWL_OFFSETOF(DD,vertices) },
    //   { "indices", OWL_BUFPTR, OWL_OFFSETOF(DD,indices) },
    //   { "texcoords", OWL_BUFPTR, OWL_OFFSETOF(DD,texcoords) },
    //   { "normals", OWL_BUFPTR, OWL_OFFSETOF(DD,normals) },
    // };
    // Geometry::addVars(params,0);
    
    // OWLModule module = owlModuleCreate
    //   (devGroup->owl,Triangles_ptx);
    // OWLGeomType gt = owlGeomTypeCreate
    //   (devGroup->owl,OWL_GEOM_TRIANGLES,sizeof(Triangles::DD),
    //    params.data(),(int)params.size());
    // owlGeomTypeSetClosestHit(gt,/*ray type*/0,module,"TrianglesCH");
    // owlGeomTypeSetAnyHit(gt,/*ray type*/0,module,"TrianglesAH");
    // owlBuildPrograms(devGroup->owl);
    // owlModuleRelease(module);
    // return gt;
    return devGroup->rtc->createTrianglesGeomType("Triangles",
                                                  sizeof(Triangles::DD),
                                                  "TrianglesAH",
                                                  "TrianglesCH");
  }

  /*! handle data arrays for vertices, indices, normals, etc; note
      that 'general' geometry attributes of the ANARI material system
      are already handled in parent class */
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
    auto devGroup = getDevGroup();
    auto rtc = devGroup->rtc;
    if (triangleGeoms.empty()) {
      rtc::GeomType *gt = devGroup->getOrCreateGeomTypeFor
        ("Triangles",Triangles::createGeomType);
      rtc::Geom *geom = rtc->createGeom(gt);
      //owlGeomCreate(getDevGroup()->owl,gt);
      triangleGeoms = { geom };
    }
    
    rtc::Geom *geom = triangleGeoms[0];
    rtc::Buffer *verticesBuffer = vertices->rtcBuffer;
    rtc::Buffer *indicesBuffer = indices->rtcBuffer;
    rtc::Buffer *texcoordsBuffer
      = texcoords
      ? texcoords->rtcBuffer
      : 0;
    rtc::Buffer *normalsBuffer
      = normals
      ? normals->rtcBuffer
      : 0;

    int numVertices = vertices->count;
    int numIndices  = indices->count;
    // owlTrianglesSetVertices(geom,verticesBuffer,
    //                         numVertices,sizeof(float3),0);
    // owlTrianglesSetIndices(geom,indicesBuffer,
    //                        numIndices,sizeof(int3),0);
    geom->setVertices(verticesBuffer,numVertices);
    geom->setIndices(indicesBuffer,numIndices);
    
    // owlGeomSetBuffer(geom,"vertices",verticesBuffer);
    // owlGeomSetBuffer(geom,"indices",indicesBuffer);
    // owlGeomSetBuffer(geom,"normals",normalsBuffer);
    // owlGeomSetBuffer(geom,"texcoords",texcoordsBuffer);

    // setAttributesOn(geom);
    // getMaterial()->setDeviceDataOn(geom);

    Triangles::DD dd;
    for (auto device : rtc->devices) {
      Geometry::writeDD(dd,device);
      dd.vertices  = (vec3f*)getDD(device,vertices);
      dd.indices   = (vec3i*)getDD(device,indices);
      dd.normals   = (vec3f*)getDD(device,normals);
      dd.texcoords = (vec2f*)getDD(device,texcoords);
      // done:
      geom->setDD(&dd,device);
    }
    
  }
  
}

