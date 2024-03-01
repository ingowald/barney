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
  std::cout << "barney unit test: going to create, build, and release random triangle mesh groups for " << numSecondsToRun << " seconds ..." << std::endl;

  int numIterations = 0;
  int numMeshesCreated = 0;
  int numTrianglesCreated = 0;
  int numGroupsBuilt = 0;
  
  while (getCurrentTime() - t0 < numSecondsToRun) {
    BNModel model = bnModelCreate(ctx);
    int numMeshes = randomInt(1,5);
    std::vector<BNGeom> geoms;
    for (int j=0;j<numMeshes;j++) {

      int numTris = randomInt(16,2000);

      std::vector<vec3f> vertices;
      std::vector<vec3i> indices;
      std::vector<vec3f> normals;
      std::vector<vec2f> texcoords;

      bool haveTex = randomInt(1);
      bool haveNor = randomInt(1);
    
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
      BNMaterialHelper mat;
      BNGeom mesh = bnTriangleMeshCreate
        (model,0,&mat,
         (int3*)indices.data(),(int)indices.size(),
         (float3*)vertices.data(),(int)vertices.size(),
         haveNor?(float3*)normals.data():nullptr,
         haveTex?(float2*)texcoords.data():nullptr);
      numMeshesCreated++;
      numTrianglesCreated += (int)indices.size();
      geoms.push_back(mesh);
    }

    BNGroup group
      = bnGroupCreate(model,0,geoms.data(),(int)geoms.size(),0,0);
    for (auto mesh : geoms) 
      bnRelease(mesh);
    
    bnGroupBuild(group);
    numGroupsBuilt ++;
    
    bnRelease(group);
    bnRelease(model);
    numIterations++;
  }

  std::cout << "done a total of " << numIterations << " create/release cycles" << std::endl;
  PRINT(numMeshesCreated);
  PRINT(numTrianglesCreated);
  PRINT(numGroupsBuilt);
  bnContextDestroy(ctx);
}

