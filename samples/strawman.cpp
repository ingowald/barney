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

#ifdef SM_USE_MPI
# include <mpi.h>
# include "barney/MPIWrappers.h"
# define SM_MPI(a) a
#else
# define SM_MPI(a)
#endif
#include "barney.h"
#include "miniScene/Scene.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION 1
#include "stb/stb_image_write.h"

using namespace mini;

void createStrawmanGeometry(BNDataGroup dataGroup,
                            int rank, int size)
{
  std::vector<BNGroup> groups;
  int gridSize = int(ceilf(2+powf(2*size,1.f/3.f)));
  if (rank == 0)
    std::cout << "#bn.sm: picking test cube grid size of "
              << gridSize << "^3" << std::endl;
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
        
        BNGeom mesh
          = bnTriangleMeshCreate(dataGroup,
                                 &material,
                                 (int3*)indices.data(),indices.size(),
                                 (float3*)vertices.data(),vertices.size(),
                                 nullptr,
                                 nullptr);
        BNGroup group
          = bnGroupCreate(dataGroup,
                          /* geometry */
                          &mesh,1,
                          /* volumes */
                          nullptr,0);
      }

  bnModelSetInstances(dataGroup,
                      groups.data(),groups.size(),
                      /* instances */
                      nullptr,0);
  // return bnModelCreate(ctx,0,
  //                      /* groups */
  //                      groups.data(),groups.size(),
  //                      /* instances */
  //                      nullptr,0);
}
  
int main(int ac, char **av)
{
  /* "APP" inits ts own MPI stuff here ... */
  /*! query which GPU(s) barney suggests to use for given rank */
#if SM_USE_MPI
  barney::mpi::init(ac,av);
  barney::mpi::Comm world(MPI_COMM_WORLD);
  
  BNHardwareInfo hardware;
  bnMPIQueryHardware(&hardware,world.comm);
  printf("#bn.sm(%i): host has %i GPUs, %i ranks on this host, and %i GPUs/rank\n",
         world.rank,
         hardware.numGPUsThisHost,
         hardware.numRanksThisHost,
         hardware.numGPUsThisRank);
  /*! which data group(s) to use in the context we are going to
    create. for this exapmle we'll create a distributed context,
    where each rank contains exactly one data group */
  std::vector<int> dataGroupsOnThisRank = { world.rank };
  /*! create the context */
  BNContext ctx = bnMPIContextCreate
    (/* the ring that binds them all...*/world.comm,
     /* what kind of data for this model _we_ own */
     dataGroupsOnThisRank.data(),dataGroupsOnThisRank.size(),
     /* GPUs we are going to use */
     nullptr,0
     );
  world.barrier();
#else
  std::vector<int> dataGroupsOnThisRank;
  for (int i=0;i<SM_NUM_LOCAL_DATA;i++)
    dataGroupsOnThisRank.push_back(i);
  BNContext ctx = bnContextCreate
    (/* what kind of data for this model _we_ own */
     dataGroupsOnThisRank.data(),dataGroupsOnThisRank.size(),
     /* GPUs we are going to use */
     nullptr,0);
#endif
  
  

  /* which data groups we will have on this rank - in this simple mpi
     sample, each rank has exactly one piece of the data */
  BNModel model
    = bnModelCreate(ctx);

#if SM_USE_MPI
  createStrawmanGeometry(bnGetDataGroup(model,world.rank),
                         world.rank,world.size);
  world.barrier();
#else
  for (auto dataID : dataGroupsOnThisRank) 
    createStrawmanGeometry(bnGetDataGroup(model,dataID),
                           dataID,dataGroupsOnThisRank.size());
#endif

  /* finalize geometry */
  bnModelBuild(model);
  SM_MPI(world.barrier());

  vec2i fbSize(800,600);
  BNFrameBuffer fb
    = bnFrameBufferCreate(ctx,
                          fbSize.x,fbSize.y);
  
  
  std::vector<uint32_t> hostFB(fbSize.x*fbSize.y);
  
  BNCamera camera;
  bnPinholeCamera(&camera,
                  /*from*/-3,-1,-2,
                  /* at */0,0,0,
                  /* up */0,1,0,
                  /*fovy*/BN_FOVY_DEGREES(30),
                  /*aspt*/0.f);

  uint32_t *fbPointer
    = 
#if SM_USE_MPI
    (world.rank>0)?nullptr:
#endif
    hostFB.data();
  
  bnRender(model,&camera,fb,fbPointer,           
           /* &renderRequest */nullptr);
  
  if (fbPointer)
    {
      std::cout << "saving rendered image..." << std::endl;
      std::string fileName = "strawman.png";
      const uint32_t *fb
        = (const uint32_t*)fbPointer;

      std::vector<uint32_t> pixels;
      for (int y=0;y<fbSize.y;y++) {
        const uint32_t *line = fb + (fbSize.y-1-y)*fbSize.x;
        for (int x=0;x<fbSize.x;x++) {
          pixels.push_back(line[x] | (0xff << 24));
        }
      }
      stbi_write_png(fileName.c_str(),fbSize.x,fbSize.y,4,
                     pixels.data(),fbSize.x*sizeof(uint32_t));
    }
  
  /* do *something* with that context .... later */
  bnContextDestroy(ctx);

#if SM_USE_MPI
  /*! "APP" closes out */
  barney::mpi::finalize();
#endif
}


