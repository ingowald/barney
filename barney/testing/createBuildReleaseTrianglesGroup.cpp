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

#include "barney.h"
#include <vector>
#include "unitTests.h"


int main(int, char **)
{
  BNContext ctx = bnContextCreate();
  double t0 = getCurrentTime();

  int numSecondsToRun = 30;

  while (getCurrentTime() - t0 < numSecondsToRun) {
    BNModel model = bnModelCreate(ctx);

    BNFrameBuffer fb = bnFrameBufferCreate(ctx,0);

    int numTris = 16 + random() % 1024;

    std::vector<vec3f> vertices;
    std::vector<vec3i> indices;
    std::vector<vec3f> normals;
    std::vector<vec2f> texcoords;

    bool haveTex = random() % 2;
    bool haveNor = random() % 2;

    for (int i=0;i<numTris;i++) {
      vec3f pos = 100.f*random3f();

      indices.push_back((int)vertices.size() + vec3i(0,1,2));
      
      vertices.push_back(pos);
      vertices.push_back(pos+random3f());
      vertices.push_back(pos+random3f());
                        
      if (haveTex) {
        texcoords.push_back(random2f());
        texcoords.push_back(random2f());
        texcoords.push_back(random2f());
      }
      
      if (haveNor) {
        normals.push_back(random3f());
        normals.push_back(random3f());
        normals.push_back(random3f());
      }
    }
    BNMaterial mat = BN_DEFAULT_MATERIAL;

    BNDataGroup dg = bnGetDataGroup(model,0);
    BNGeom mesh = bnTriangleMeshCreate
      (dg,&mat,
       (int3*)indices.data(),indices.size(),
       (float3*)vertices.data(),vertices.size(),
       haveNor?(float3*)normals.data():nullptr,
       haveTex?(float2*)texcoords.data():nullptr);
    
    bnRelease(mesh);
    bnRelease(model);
  }

  bnContextDestroy(ctx);
}

