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

#include "barney/Triangles.h"
#include "barney/DataGroup.h"

namespace barney {

  extern "C" char Triangles_ptx[];

  Triangles::Triangles(DataGroup *owner,
            const Material &material,
            int numIndices,
            const vec3i *indices,
            int numVertices,
            const vec3f *vertices,
            const vec3f *normals,
            const vec2f *texcoords)
    : Geometry(owner,material)
  {
    OWLGeomType gt = owner->devGroup->getOrCreateGeomTypeFor
      ("Triangles",Triangles::createGeomType);
    OWLGeom geom = owlGeomCreate(owner->devGroup->owl,gt);

    verticesBuffer = owlDeviceBufferCreate
      (owner->devGroup->owl,
       OWL_FLOAT3,numVertices,vertices);
    indicesBuffer = owlDeviceBufferCreate
      (owner->devGroup->owl,
       OWL_INT3,numIndices,indices);

    owlTrianglesSetVertices(geom,verticesBuffer,
                            numVertices,sizeof(float3),0);
    owlTrianglesSetIndices(geom,indicesBuffer,
                           numIndices,sizeof(int3),0);
    owlGeomSetRaw(geom,"material",&material);
    owlGeomSetBuffer(geom,"vertices",verticesBuffer);
    owlGeomSetBuffer(geom,"indices",indicesBuffer);
    
    triangleGeoms.push_back(geom);
  }

  void Triangles::update(const Material &material,
                         int numIndices,
                         const vec3i *indices,
                         int numVertices,
                         const vec3f *vertices,
                         const vec3f *normals,
                         const vec2f *texcoords)
  {
    OWLGeom geom = triangleGeoms[0];
    
    owlBufferResize(verticesBuffer,numVertices);
    owlBufferResize(indicesBuffer,numIndices);
    owlBufferUpload(verticesBuffer,vertices);
    owlBufferUpload(indicesBuffer,indices);

    owlTrianglesSetVertices(geom,verticesBuffer,
                            numVertices,sizeof(float3),0);
    owlTrianglesSetIndices(geom,indicesBuffer,
                           numIndices,sizeof(int3),0);
    owlGeomSetRaw(geom,"material",&material);
    owlGeomSetBuffer(geom,"vertices",verticesBuffer);
    owlGeomSetBuffer(geom,"indices",indicesBuffer);
  }
  
  OWLGeomType Triangles::createGeomType(DevGroup *devGroup)
  {
    std::cout << OWL_TERMINAL_GREEN
              << "creating 'Triangles' geometry type"
              << OWL_TERMINAL_DEFAULT << std::endl;
    
    static OWLVarDecl params[]
      = {
         { "material", OWL_USER_TYPE(Material), OWL_OFFSETOF(DD,material) },
         { "vertices", OWL_BUFPTR, OWL_OFFSETOF(DD,vertices) },
         { "indices", OWL_BUFPTR, OWL_OFFSETOF(DD,indices) },
         { nullptr }
    };
    OWLModule module = owlModuleCreate
      (devGroup->owl,Triangles_ptx);
    OWLGeomType gt = owlGeomTypeCreate
      (devGroup->owl,OWL_GEOM_TRIANGLES,sizeof(Triangles::DD),
       params,-1);
    owlGeomTypeSetClosestHit(gt,/*ray type*/0,module,"TrianglesCH");
    owlBuildPrograms(devGroup->owl);
    
    return gt;
  }
  
}

