// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#if !BARNEY_MPI
# error "this file should only ever be included when BARNEY_MPI support is enabled"
#endif

#include "Device.h"
#include <mpi.h>

namespace BARNEY_NS {
  namespace anari {

    struct BarneyMPIDevice : public BarneyDevice {
      BarneyMPIDevice();
      BarneyMPIDevice(ANARILibrary library, const std::string &subType = "default");
      ~BarneyMPIDevice() override;
    
      BNContext createContext(std::vector<vec2i> &gpuIDsAndDataRank);
      void initMPI() override;
      void deviceCommitParameters() override;
    
      /*! communicator to use for barney data-parallel rendering, set as
        a uint64_t. If set to 0, we'll use local rendering even if mpi
        support is compiled in, any other value will be interpreted as
        a MPI_Comm type. If device gets created with subtype "mpi" or
        "default", the default value for comm is MPI_COMM_WORLD, if it
        is created with subtype "local" it will default to 0 */
      MPI_Comm comm = 0;//MPI_COMM_WORLD;
      bool     commNeedsFree = false;
    };
  
  }
}

