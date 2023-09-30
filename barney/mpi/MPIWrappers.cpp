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

#include "barney/mpi/MPIWrappers.h"

namespace barney {
  namespace mpi {

    void init(int &ac, char **av)
    {
      int required = MPI_THREAD_MULTIPLE;
      int provided = 0;
      BN_MPI_CALL(Init_thread(&ac,&av,required,&provided),"could not init mpi");
      if (provided != required)
        throw barney::mpi::Exception(__PRETTY_FUNCTION__,-1,
                                     "MPI runtime does not provide threading support");
    }

    void finalize()
    {
      BN_MPI_CALL(Finalize(),"error in mpi::finalize");
    }

    Comm::Comm(MPI_Comm comm)
      : comm(comm)
    {
      if (comm != MPI_COMM_NULL) {
        BN_MPI_CALL(Comm_rank(comm,&rank),"cannot query comm rank");
        BN_MPI_CALL(Comm_size(comm,&size),"cannot query comm size");
      }
    }

    void Comm::assertValid() const
    {
      if (comm == MPI_COMM_NULL)
        throw barney::mpi::Exception(__PRETTY_FUNCTION__,-1,
                                     "not a valid mpi communicator"); 
    }

    int Comm::allReduceMax(int value) const
    {
      int result = 0;
      BN_MPI_CALL(Allreduce(&value,&result,1,MPI_INT,MPI_MAX,comm),
                  "could not compute mpi reduce-max");
      return result;
    }
    
    int Comm::allReduceMin(int value) const
    {
      int result = 0;
      BN_MPI_CALL(Allreduce(&value,&result,1,MPI_INT,MPI_MIN,comm),
                  "could not compute mpi reduce-min");
      return result;
    }
    
    void Comm::barrier() const
    {
      BN_MPI_CALL(Barrier(comm),
                  "error in mpi-barrier");
    }
  }
}
