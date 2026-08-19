// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "Device.h"
#include "MPIDevice.h"
#include <mpi.h>

#include "Array.h"
#include "Frame.h"
// std
#include <cstring>

#include "barney/native/include/barney_mpi.h"


namespace BARNEY_NS {
  namespace native {
    size_t getHostNameHash();
  }
  namespace anari {

    /*! checks whether all the ranks on the communicator are on
        different physical hosts (based on hostname */
    bool allOnDifferentHosts(MPI_Comm comm)
    {
      int size = 0;
      MPI_Comm_size(comm,&size);
      size_t myHash = native::getHostNameHash();
      std::vector<size_t> allHashes(size);
      MPI_Allgather(&myHash,
                    1,MPI_LONG_LONG,allHashes.data(),
                    1,MPI_LONG_LONG,comm);
      std::map<size_t,bool> alreadyFound;
      for (auto hash: allHashes) {
        if (alreadyFound[hash]) return false;
        alreadyFound[hash] = true;
      }
      return true;
    }
    
    BarneyMPIDevice::BarneyMPIDevice()
      : BarneyDevice()
    {
      m_enable_multiGPU = 0;
      PING; PRINT(m_enable_multiGPU);
    }
  
    BarneyMPIDevice::BarneyMPIDevice(ANARILibrary library,
                                     const std::string &subType)
      : BarneyDevice(library,subType)
    {
      PING;
      m_enable_multiGPU = 0;
    }
  
    BarneyMPIDevice::~BarneyMPIDevice()
    {
      if (commNeedsFree)
        MPI_Comm_free(&comm);
    }
  
    void BarneyMPIDevice::initMPI() 
    {
      assert(comm
             && "BarneyMPIDevice - comm not set!?");
    
      int mpiInitialized = 0;
      MPI_Initialized(&mpiInitialized);

      if (!mpiInitialized) {
        std::cout << "#barney: anari_barney device created in MPI mode (loaded from barney_mpi device in either mpi or default mode), but MPI not yet initialized. Doing so now, but this is not how it should be."
                  << std::endl;
        int required = MPI_THREAD_MULTIPLE;
        int provided = 0;
        MPI_Init_thread(nullptr,nullptr,required,&provided);
      }

      int rank,size;
      MPI_Comm_rank(comm, &rank);
      MPI_Comm_size(comm, &size);
      printf("#banari.mpi: running banari mpi device on rank %i size %i\n",
             rank,size);
      multiNode.rank = rank;
      multiNode.size = size;
    }
  
    void BarneyMPIDevice::deviceCommitParameters()
    {
      /* we do ours FIRST, so BarneyDevice::commit() can then directly
         create the device; if we called parent commit first the device
         would be created without this comm being set. */
      uint64_t pointerToComm = getParam<uint64_t>("pointer_to_mpi_communicator", 0ull);
      if (pointerToComm) {
        printf("#banari.mpi: got passed a pointer to a MPI "
               "communicator, going to use this.\n");
        comm = *(MPI_Comm *)pointerToComm;
        commNeedsFree = false;
        assert(comm
               && "BarneyMPI: app did set comm, but set it to a null communicator?");
      } else {
        std::cout << "#banari: Device started in MPI mode, but no MPI Communicator passed to it; started with a MPI_Comm_dup() of MPI_COMM_WORLD" << std::endl;
        MPI_Comm_dup(MPI_COMM_WORLD, &comm);
        commNeedsFree = true;
      }
      // if (comm) {
      //   int rank, size;
      //   MPI_Comm_rank(comm,&rank);
      //   MPI_Comm_size(comm,&size);
      // }

      BarneyDevice::deviceCommitParameters();
    }

    BNContext BarneyMPIDevice::createContext(const std::vector<int> &dataRanks,
                                             const std::vector<int> &gpuIDs)
    {
      BNContext ctx =
        bnMPIContextCreate(comm,
                           dataRanks.size(),
                           dataRanks.data(),
                           gpuIDs.size(),
                           gpuIDs.data());
      return ctx;
    }

  }
}
