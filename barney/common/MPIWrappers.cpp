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

#include "barney/common/MPIWrappers.h"

namespace barney_api {
  namespace mpi {

    void init(int &ac, char **av)
    {
      int required = MPI_THREAD_MULTIPLE;
      int provided = 0;
      BN_MPI_CALL(Init_thread(&ac,&av,required,&provided));
      if (provided != required)
        throw std::runtime_error("MPI runtime does not provide threading support");
    }

    void finalize()
    {
      BN_MPI_CALL(Finalize());
    }

    Comm::Comm(MPI_Comm comm)
      : comm(comm)
    {
      if (comm != MPI_COMM_NULL) {
        BN_MPI_CALL(Comm_rank(comm,&rank));
        BN_MPI_CALL(Comm_size(comm,&size));
      }
    }

    /*! master's send side of broadcast - must be done on rank 0,
      and matched by bc_recv on all workers */
    void Comm::bc_send(const void *data, size_t numBytes)
    {
      BN_MPI_CALL(Bcast((void *)data,numBytes,MPI_BYTE,0,comm));
    }
    
    /*! receive side of a broadcast - must be called on all ranks >
      0, and match a bc_send on rank 0 */
    void Comm::bc_recv(void *data, size_t numBytes)
    {
      BN_MPI_CALL(Bcast(data,numBytes,MPI_BYTE,0,comm));
    }
    
    void Comm::assertValid() const
    {
      if (comm == MPI_COMM_NULL)
        throw std::runtime_error(std::string(__PRETTY_FUNCTION__)
                                 +" : not a valid mpi communicator"); 
    }
    
    int Comm::allReduceAdd(int value) const
    {
      int result;
      BN_MPI_CALL(Allreduce(&value,&result,1,MPI_INT,MPI_SUM,comm));
      return result;
    }

    float Comm::allReduceAdd(float value) const
    {
      float result;
      BN_MPI_CALL(Allreduce(&value,&result,1,MPI_FLOAT,MPI_SUM,comm));
      return result;
    }

    int Comm::allReduceMin(int value) const
    {
      int result;
      BN_MPI_CALL(Allreduce(&value,&result,1,MPI_INT,MPI_MIN,comm));
      return result;
    }

    int Comm::allReduceMax(int value) const
    {
      int result;
      BN_MPI_CALL(Allreduce(&value,&result,1,MPI_INT,MPI_MAX,comm));
      return result;
    }
    
    float Comm::allReduceMax(float value) const
    {
      float result;
      BN_MPI_CALL(Allreduce(&value,&result,1,MPI_FLOAT,MPI_MAX,comm));
      return result;
    }
    
    float Comm::allReduceMin(float value) const
    {
      float result;
      BN_MPI_CALL(Allreduce(&value,&result,1,MPI_FLOAT,MPI_MIN,comm));
      return result;
    }

    vec3f Comm::allReduceMin(vec3f v) const
    {
      return vec3f(allReduceMin(v.x),allReduceMin(v.y),allReduceMin(v.z));
    }
    
    vec3f Comm::allReduceMax(vec3f v) const
    {
      return vec3f(allReduceMax(v.x),allReduceMax(v.y),allReduceMax(v.z));
    }
    
    void Comm::allGather(int *allValues, int myValue)
    {
      BN_MPI_CALL(Allgather(&myValue,1,MPI_INT,allValues,1,MPI_INT,comm));
    }
    
    void Comm::allGather(int *allValues, const int *myValues, int numMyValues)
    {
      BN_MPI_CALL(Allgather(myValues,numMyValues,MPI_INT,allValues,numMyValues,MPI_INT,comm));
    }
    
    /*! free/close this communicator */
    void Comm::free()
    {
      BN_MPI_CALL(Comm_free(&comm));
    }
      
    /*! equivalent of MPI_Comm_split - splits this comm into
      possibly multiple comms, each one containing exactly those
      former ranks of the same color. E.g. split(old.rank > 0)
      would have rank 0 get a communicator that contains only
      itself, and all others get a communicator that contains all
      other former ranks */
    Comm Comm::split(int color)
    {
      MPI_Comm newComm;
      BN_MPI_CALL(Comm_split(comm,color,rank,&newComm));
      return Comm(newComm);
    }
      

    
    void Comm::barrier() const
    {
      BN_MPI_CALL(Barrier(comm));
    }
  }
}
