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

#include <mpi.h>
#include "barney.h"
#include "barney/mpi/MPIWrappers.h"
#include "miniScene/Scene.h"

using namespace mini;

BNModel createStrawmanModel(BNContext ctx,
                         BNModel model,
                         int rank, int size)
{
  std::vector<BNGroup> groups;
  int gridSize = int(ceilf(2+powf(2*size,1.f/3.f)));
  for (int iz=0;iz<gridSize;iz++)
    for (int iy=0;iy<gridSize;iy++)
      for (int ix=0;ix<gridSize;ix++) {
        int boxID = ix + gridSize*(iy + gridSize * (iz));
        if (boxID % size != rank) continue;
        
        vec3f lower = vec3i(ix,iy,iz) + .1f;
        vec3f upper = lower + .8f;

        lower = lower * (2.f/gridSize) - 1.f;
        upper = upper * (2.f/gridSize) - 1.f;

        std::vector<vec3f> vertices;

        vertices.push_back(vec3f(lower.x,lower.y,lower.z));
        vertices.push_back(vec3f(upper.x,lower.y,lower.z));
        vertices.push_back(vec3f(lower.x,upper.y,lower.z));
        vertices.push_back(vec3f(upper.x,upper.y,lower.z));
        vertices.push_back(vec3f(lower.x,lower.y,upper.z));
        vertices.push_back(vec3f(upper.x,lower.y,upper.z));
        vertices.push_back(vec3f(lower.x,upper.y,upper.z));
        vertices.push_back(vec3f(upper.x,upper.y,upper.z));

        std::vector<vec3i> indices
          = {{0,1,3}, {2,3,0},
             {5,7,6}, {5,6,4},
             {0,4,5}, {0,5,1},
             {2,3,7}, {2,7,6},
             {1,5,7}, {1,7,3},
             {4,0,2}, {4,2,6}
        };

        BNMaterial material = BN_DEFAULT_MATERIAL;
        float r = ((0x123+13*19*boxID) % 256) / 255.f;
        float g = ((0x123+17*23*boxID) % 256) / 255.f;
        float b = ((0x123+27*37*boxID) % 256) / 255.f;
        (vec3f&)material.diffuseColor = vec3f(r,g,b);
        
        BNGeom mesh = bnTriangleMeshCreate(ctx,0,
                                           &material,
                                           (int3*)indices.data(),indices.size(),
                                           (float3*)vertices.data(),vertices.size(),
                                           nullptr,
                                           nullptr);
        BNGroup group = bnGroupCreate(ctx,0,
                                      /* geometry */
                                      &mesh,1,
                                      /* volumes */
                                      nullptr,0);
      }
  return bnModelCreate(ctx,0,
                       /* groups */
                       groups.data(),groups.size(),
                       /* instances */
                       nullptr,0);
}
  
int main(int ac, char **av)
{
  /* "APP" inits ts own MPI stuff here ... */
  barney::mpi::init(ac,av);
  barney::mpi::Comm world(MPI_COMM_WORLD);

  PRINT(world.rank);
  PRINT(world.size);

  /*! query which GPU(s) barney suggests to use for given rank */
  BNHardwareInfo hardware;
  bnQueryHardwareMPI(&hardware,world.comm);
  printf("#bn.sm(%i): node has %i GPUs, %i ranks on this node, and %i GPUs/rank\n",
         world.rank,hardware.gpusOnNode,hardware.ranksOnNode,hardware.gpusOnRank);

  /*! which data group(s) to use in the context we are going to
    create. for this exapmle we'll create a distributed context,
    where each rank contains exactly one data group */
  std::vector<int> dataGroupsOnThisRank = { world.rank };
  
  /*! create the context */
  BNContext ctx = bnContextCreateMPI
    (/* the ring that binds them all...*/world.comm,
     /* what kind of data for this model _we_ own */
     dataGroupsOnThisRank.data(),dataGroupsOnThisRank.size(),
     /* GPUs we are going to use */
     nullptr,0
     );

  /* which data groups we will have on this rank - in this simple mpi
     sample, each rank has exactly one piece of the data */
  BNModel model
    = createStrawmanModel(ctx,model,world.rank,world.size);

  /* do *something* with that context .... later */
  bnContextDestroy(ctx);

  
  /*! "APP" closes out */
  barney::mpi::finalize();
}


