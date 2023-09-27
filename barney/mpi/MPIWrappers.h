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

#include "barney/core/common/common.h"
#include <mpi.h>
#include <stdexcept>

namespace barney {
  namespace mpi {

    struct Exception : public std::runtime_error {
      Exception(const std::string &where, int rc, const std::string &msg)
        : std::runtime_error("#barney.mpi (@"+where+") : " + msg)
      {}
    };
    
    void init(int &ac, char **av);
    void finalize();
    
    struct Comm {
      Comm(MPI_Comm comm);

      inline operator MPI_Comm() { return comm; }
      void assertValid();
      int allReduceMax(int value);
      
      int rank = -1, size = -1;
      MPI_Comm comm = MPI_COMM_NULL;
    };
    
  }
}
