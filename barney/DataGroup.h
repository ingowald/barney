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

#include "barney/Group.h"

namespace barney {

  struct Model;
  struct Spheres;
  struct Triangles;
  
  struct DataGroup : public Object {
    typedef std::shared_ptr<DataGroup> SP;

    DataGroup(Model *model, int localID);
    
    static SP create(Model *model, int localID)
    { return std::make_shared<DataGroup>(model,localID); }

    Group   *createGroup(const std::vector<Geometry::SP> &geoms);
    Spheres *createSpheres(const barney::Material &material,
                           const vec3f *origins,
                           int numOrigins,
                           const float *radii,
                           float defaultRadius);
    Triangles *createTriangles(const barney::Material &material,
                               int numIndices,
                               const vec3i *indices,
                               int numVertices,
                               const vec3f *vertices,
                               const vec3f *normals,
                               const vec2f *texcoords);
    void setInstances(std::vector<Group::SP> &groups,
                      const affine3f *xfms);
    
    struct {
      std::vector<Group::SP> groups;
      std::vector<affine3f>  xfms;
      OWLGroup group = 0;
    } instances;

    void build();

    DevGroup::SP const devGroup;
    Model       *const model;
    int          const localID;
  };
  
}
