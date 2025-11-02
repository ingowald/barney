// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#include "barney/geometry/Triangles.h"
#include "barney/ModelSlot.h"
#include "barney/Context.h"

namespace BARNEY_NS {

  RTC_IMPORT_TRIANGLES_GEOM(Triangles,Triangles,Triangles::DD,true,false);

  Triangles::Triangles(Context *context, DevGroup::SP devices)
    : Geometry(context,devices)
  {}
  
  Triangles::~Triangles()
  {}
  
  /*! handle data arrays for vertices, indices, normals, etc; note
      that 'general' geometry attributes of the ANARI material system
      are already handled in parent class */
  bool Triangles::setData(const std::string &member,
                          const barney_api::Data::SP &value)
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
    for (auto device : *devices) {
      PLD *pld = getPLD(device);
      if (pld->triangleGeoms.empty()) {
        rtc::GeomType *gt
          = device->geomTypes.get(createGeomType_Triangles);
        rtc::Geom *geom = gt->createGeom();
        pld->triangleGeoms = { geom };
      }
    
      rtc::Geom *geom = pld->triangleGeoms[0];
      rtc::Buffer *verticesBuffer
        = vertices->getPLD(device)->rtcBuffer;
      rtc::Buffer *indicesBuffer
        = indices->getPLD(device)->rtcBuffer;
      
      int numVertices = (int)vertices->count;
      int numIndices  = (int)indices->count;
      
      geom->setVertices(verticesBuffer,numVertices);
      geom->setIndices(indicesBuffer,numIndices);
      Triangles::DD dd;
      Geometry::writeDD(dd,device);
      dd.vertices  = (vec3f*)vertices->getDD(device);
      dd.indices   = (vec3i*)indices->getDD(device);
      dd.normals   = (vec3f*)(normals?normals->getDD(device):0);
      dd.texcoords = (vec2f*)(texcoords?texcoords->getDD(device):0);

      // done:
      geom->setDD(&dd);
    }
    
  }
}

