// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "barney.h"
#include <mpi.h>

BEGIN_BARNEY_NS

BARNEY_API
BNContext bnMPIContextCreate(MPI_Comm comm,
                             /*! this is the NUMBER of different data
                               groups/slots inthat context. a data
                               *group* is one or more GPUs that share a
                               common set of geometric/volumetric
                               objects. */
                             int        numDataGroupsOnThisContext,
                             /*! tells which data rank / 'color' of data
                               will be stored in each of the
                               `numDataGroupsOnThisContext` data groups
                               of this context. This array *must* have
                               `numDataGroupsOnThisContext` entries */
                             const int *dataRanksInDataGroup,
                             /*! which gpu(s) to use for this
                               process. GPUs will be assigned to data
                               groups on a round-robin basis, so the i'th
                               GPU listed here will get assigned to data
                               group 'i%numDataGroupsOnThisContext' */
                             int numGPUs,
                             const int *gpuIDs);


END_BARNEY_NS
