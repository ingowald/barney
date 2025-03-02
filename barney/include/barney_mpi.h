// ======================================================================== //
// Copyright 2023-2025 Ingo Wald                                            //
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

#include "barney.h"
#include <mpi.h>

BARNEY_API
BNContext bnMPIContextCreate(MPI_Comm comm,
                             /*! how many data slots this context is to
                               offer, and which part(s) of the
                               distributed model data these slot(s)
                               will hold */
                             const int *dataRanksOnThisContext=0,
                             int        numDataRanksOnThisContext=1,
                             /*! which gpu(s) to use for this
                               process. default is to distribute
                               node's GPUs equally over all ranks on
                               that given node */
                             const int *gpuIDs=nullptr,
                             int  numGPUs=-1
                             );

BARNEY_API
void  bnMPIQueryHardware(BNHardwareInfo *hardware, MPI_Comm comm);

