// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


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

// BARNEY_API
// void  bnMPIQueryHardware(BNHardwareInfo *hardware, MPI_Comm comm);

