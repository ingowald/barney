// ======================================================================== //
// Copyright 2023-2025 Ingo Wald                                            //
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

#include "rtcore/optix/Geom.h"
#include "rtcore/optix/Buffer.h"
#include "rtcore/optix/Device.h"

namespace rtc {
  namespace optix {

    // ==================================================================
    // geom
    // ==================================================================

    Geom::Geom(GeomType *gt,
               OWLGeom geom)
      : gt(gt),
        owl(geom)
    {}
    
    Geom::~Geom()
    {
      owlGeomRelease(owl);
    }

    void Geom::setDD(const void *dd)
    {
      owlGeomSetRaw(owl,"raw",dd);
    }

    TrianglesGeom::TrianglesGeom(GeomType *gt,
                                 OWLGeom geom)
      : Geom(gt,geom)
    {}
    UserGeom::UserGeom(GeomType *gt,
                       OWLGeom geom)
      : Geom(gt,geom)
    {}
    
    /*! only for user geoms */
    void UserGeom::setPrimCount(int primCount)
    {
      owlGeomSetPrimCount(owl,primCount);
    }
    
    /*! can only get called on triangle type geoms */
    void TrianglesGeom::setVertices(Buffer *vertices, int numVertices)
    {
      owlTrianglesSetVertices(owl,((Buffer*)vertices)->owl,
                              numVertices,sizeof(float3),0);
    }
    
    void TrianglesGeom::setIndices(Buffer *indices, int numIndices)
    {
      owlTrianglesSetIndices(owl,((Buffer*)indices)->owl,
                             numIndices,sizeof(int3),0);
    }

    GeomType::GeomType(optix::Device *device)
      : device(device)
    {}
    
    GeomType::~GeomType()
    {
      // CANNOYT yet release this because owl cannot do that yet
      gt = 0;
    }

    TrianglesGeomType::TrianglesGeomType(optix::Device *device,
                                         const std::string &ptxCode,
                                         const std::string &typeName,
                                         size_t sizeOfDD,
                                         bool has_ah, bool has_ch)
      : GeomType(device)
    {
      OWLVarDecl vars[] = {
        {"raw",(OWLDataType)(OWL_USER_TYPE_BEGIN+sizeOfDD),0},
        {nullptr}
      };
      gt = owlGeomTypeCreate(device->owl,OWL_GEOM_TRIANGLES,
                             sizeOfDD,vars,-1);
      
      const char *ptx = ptxCode.c_str();
      // = (const char *)rtc::getSymbol(ptxName);//+"_ptx");
      OWLModule module = owlModuleCreate
        (device->owl,ptx);
      if (has_ch)
        owlGeomTypeSetClosestHit(gt,/*ray type*/0,module,
                                 typeName.c_str());
      // chFctName.c_str());//"TrianglesCH");
      if (has_ah)
        owlGeomTypeSetAnyHit(gt,/*ray type*/0,module,
                             typeName.c_str());
      // ahFctName.c_str());//"TrianglesAH");
      owlBuildPrograms(device->owl);
      owlModuleRelease(module);
    }

    UserGeomType::UserGeomType(optix::Device *device,
                               const std::string &ptxCode,
                               const std::string &typeName,
                               size_t sizeOfDD,
                               bool has_ah, bool has_ch)
      : GeomType(device)
    {
      OWLVarDecl vars[] = {
        {"raw",(OWLDataType)(OWL_USER_TYPE_BEGIN+sizeOfDD),0},
        {nullptr}
      };
      gt = owlGeomTypeCreate(device->owl,OWL_GEOM_USER,
                             sizeOfDD,vars,-1);
      
      const char *ptx = ptxCode.c_str();

      OWLModule module = owlModuleCreate
        (device->owl,ptx);
      if (has_ch)
        owlGeomTypeSetClosestHit(gt,/*ray type*/0,module,
                                 typeName.c_str());
      if (has_ah)
        owlGeomTypeSetAnyHit(gt,/*ray type*/0,module,
                             typeName.c_str());
      owlGeomTypeSetBoundsProg(gt,/*ray type*/module,
                               typeName.c_str());
      owlGeomTypeSetIntersectProg(gt,/*ray type*/0,module,
                                  typeName.c_str());
      owlBuildPrograms(device->owl);
      owlModuleRelease(module);
    }
    
    Geom *TrianglesGeomType::createGeom()
    {
      OWLGeom geom = owlGeomCreate(((optix::Device*)device)->owl,this->gt);
      return new TrianglesGeom(this,geom);
    }

    Geom *UserGeomType::createGeom()
    {
      OWLGeom geom = owlGeomCreate(((optix::Device*)device)->owl,this->gt);
      return new UserGeom(this,geom);
    }
      
  }
}
