// ======================================================================== //
// Copyright 2023-2024 Ingo Wald                                            //
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

#include "barney/Group.h"
#include "barney.h"
#include <set>

namespace barney {

  struct Model;
  struct Spheres;
  struct Cylinders;
  struct Triangles;
  struct Context;
  struct Texture;
  struct Data;
  struct Light;
  
  struct DataGroup : public Object {
    typedef std::shared_ptr<DataGroup> SP;

    DataGroup(Model *model, int localID);
    virtual ~DataGroup();

    /*! pretty-printer for printf-debugging */
    std::string toString() const override { return "barney::DataGroup"; }
    
    OWLContext getOWL() const;

    static SP create(Model *model, int localID)
    { return std::make_shared<DataGroup>(model,localID); }

    Group   *
    createGroup(const std::vector<Geometry::SP> &geoms,
                const std::vector<Volume::SP> &volumes);
    
    Spheres *
    createSpheres(const barney::Material &material,
                  const vec3f *origins,
                  int numOrigins,
                  const vec3f *colors,
                  const float *radii,
                  float defaultRadius);

    Cylinders *createCylinders(const Material   &material,
                               const vec3f      *points,
                               int               numPoints,
                               const vec3f      *colors,
                               bool              colorPerVertex,
                               const vec2i      *indices,
                               int               numIndices,
                               const float      *radii,
                               bool              radiusPerVertex,
                               float             defaultRadius);
    
    Volume *createVolume(ScalarField::SP sf);
    
    Triangles *
    createTriangles(const barney::Material &material,
                    int numIndices,
                    const vec3i *indices,
                    int numVertices,
                    const vec3f *vertices,
                    const vec3f *normals,
                    const vec2f *texcoords);

    Texture *
    createTexture(BNTexelFormat texelFormat,
                  vec2i size,
                  const void *texels,
                  BNTextureFilterMode  filterMode,
                  BNTextureAddressMode addressMode,
                  BNTextureColorSpace  colorSpace);
    
    ScalarField *createStructuredData(const vec3i &dims,
                                      BNScalarType scalarType,
                                      const void *data,
                                      const vec3f &gridOrigin,
                                      const vec3f &gridSpacing);
    
    ScalarField *createUMesh(std::vector<vec4f> &vertices,
                             std::vector<TetIndices> &tetIndices,
                             std::vector<PyrIndices> &pyrIndices,
                             std::vector<WedIndices> &wedIndices,
                             std::vector<HexIndices> &hexIndices,
                             std::vector<int> &gridOffsets,
                             std::vector<vec3i> &gridDims,
                             std::vector<box4f> &gridDomains,
                             std::vector<float> &gridScalars,
                             const box3f &domain);
   
    ScalarField *createBlockStructuredAMR(std::vector<box3i> &blockBounds,
                                          std::vector<int> &blockLevels,
                                          std::vector<int> &blockOffsets,
                                          std::vector<float> &blockScalars);

    Data *createData(BNDataType dataType,
                     size_t numItems,
                     const void *items);
    Light *createLight(const std::string &type);
    void setInstances(std::vector<Group::SP> &groups,
                      const affine3f *xfms);
    
    struct {
      std::vector<Group::SP> groups;
      std::vector<affine3f>  xfms;
      OWLGroup group = 0;
    } instances;

    void build();

    MultiPass::Instances multiPassInstances;
    DevGroup::SP   const devGroup;
    Model         *const model;
    int            const localID;
  };
  
}
