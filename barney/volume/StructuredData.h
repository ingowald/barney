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

#pragma once

#include "barney/DataGroup.h"
#include "barney/volume/MCAccelerator.h"

namespace barney {

  struct DataGroup;
  
  /*! a structured 3D scalar field; strictly speaking in barney this
      isn't actually a "volume" (it only becomes a volume after paired
      with a transfer function), but this name still sounds more
      reasonable thatn `Structured3DScalarField`, which would be a
      more accurate name */
  struct StructuredData : public ScalarField
  {
    struct DD : public ScalarField::DD {
      cudaTextureObject_t texObj;
      vec3f cellGridOrigin;
      vec3f cellGridSpacing;
      vec3i numCells;
      
      static void addVars(std::vector<OWLVarDecl> &vars, int base);
    };
    
    struct Tex3D {
      
      cudaArray_t           voxelArray = 0;
      cudaTextureObject_t   texObj;
      cudaTextureObject_t   texObjNN;
    };
    /*! one tex3d per device */
    std::vector<Tex3D> tex3Ds;

    StructuredData(DataGroup *owner,
                   DevGroup *devGroup,
                   const vec3i &numScalars,
                   BNScalarType scalarType,
                   const void *scalars,
                   const vec3f &gridOrigin,
                   const vec3f &gridSpacing);

    // /*! returns (part of) a string that should allow an OWL geometry
    //     type to properly create all the names of all the optix device
    //     functions that operate on this type. Eg, if all device
    //     functions for a "StucturedVolume" are named
    //     "Structured_<SomeAccel>_{Bounds,Isec,CV}()", then the
    //     StructuredField should reutrn "Structured", and somebody else
    //     can/has to then make sure to add the respective
    //     "_<SomeAccel>_" part. */
    // std::string getTypeString() const override { return "Structured"; }
    
    void setVariables(OWLGeom geom);
    // std::vector<OWLVarDecl> getVarDecls(uint32_t baseOfs) { BARNEY_NYI(); };
    VolumeAccel::SP createAccel(Volume *volume) override;
    void buildMCs(MCGrid &macroCells) override;

    void createCUDATextures();
    
    const BNScalarType   scalarType;
    const vec3i numScalars;
    const vec3i numCells;
    const vec3f gridOrigin;
    const vec3f gridSpacing;
    const void *rawScalarData;
  };

  /*! for structured data, the sampler doesn't have to do much but
      sample the 3D texture that the structeuddata field has already
      created. in thoery one could argue that the 3d texture should
      belong ot the sampler (not the field), but the field needs it to
      compute the macro cells, so we'll leave it as such for now */
  struct StructuredDataSampler {
    struct DD : public StructuredData::DD {
      inline __device__ float sample(const vec3f P, bool dbg) const
      {
        vec3f rel = (P - cellGridOrigin) * rcp(cellGridSpacing);

        // if (dbg) printf("sample %f %f %f rel %f %f %f\n",
        //                 P.x,P.y,P.z,
        //                 rel.x,rel.y,rel.z
        //                 );
        
        if (rel.x < 0.f) return NAN;
        if (rel.y < 0.f) return NAN;
        if (rel.z < 0.f) return NAN;
        if (rel.x >= numCells.x) return NAN;
        if (rel.y >= numCells.y) return NAN;
        if (rel.z >= numCells.z) return NAN;
        float f = tex3D<float>(texObj,rel.x,rel.y,rel.z);
        // if (dbg) printf("result %f\n",f);
        return f;
      }
    };

    struct Host
    {
      Host(ScalarField *field)
        : field((StructuredData *)field)
      {}

      /*! builds the string that allows for properly matching optix
        device progs for this type */
      inline std::string getTypeString() const { return "Structured"; }

      /*! doesn'ta ctualy do anything for this class, but required to
          make the template instantiating it happy */
      void setVariables(OWLGeom geom) { /* nothing to do for this class */}
      
      /*! doesn'ta ctualy do anything for this class, but required to
          make the template instantiating it happy */
      void build(bool full_rebuild) { /* nothing to do for this class */}
      
      StructuredData *const field;
    };
    
  };

}

