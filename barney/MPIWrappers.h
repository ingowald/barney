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

#include "barney/common.h"
#include <mpi.h>
#include <stdexcept>

#define BN_MPI_CALL(fctCall, err)                                                 \
    { int rc = MPI_##fctCall; if (rc != MPI_SUCCESS) throw barney::mpi::Exception(__PRETTY_FUNCTION__,rc,err); }
    
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
      void assertValid() const;
      int  allReduceMax(int value) const;
      int  allReduceMin(int value) const;
      void barrier() const;

      /*! master-side of a gather where clietn gathers a fixed number
          of itmes from each rank */
      template<typename T>
      void masterGather(// where we'll receive into - for ALL ranks
                        T *recvBuffer,
                        // what we're sending (this rank's data only)
                        const T *sendBuffer, int numItemsSentOnEachRank);
      /*! client-side of a gather where each client send a fixed number
          of items to the master */
      template<typename T>
      void masterGather(// what we're sending (this rank's data only)
                        const T *sendBuffer, int numItemsSentOnEachRank);

      template<typename T>
      void recv(int fromRank, int tag,
                T *buffer, int numItems, MPI_Request &req);

      template<typename T>
      void send(int fromRank, int tag,
                const T *buffer, int numItems, MPI_Request &req);

      void wait(MPI_Request &req)
      {
        BN_MPI_CALL(Wait(&req,MPI_STATUS_IGNORE),"mpi-wait");
      }
      
      int rank = -1, size = -1;
      MPI_Comm comm = MPI_COMM_NULL;
    };
    
    template<typename T>
    inline void Comm::recv(int fromRank, int tag,
                                  T *buffer, int numItems, MPI_Request &req)
    {
      // BN_MPI_CALL(Recv(buffer,numItems*sizeof(T),MPI_BYTE,
      //                  fromRank,tag,comm,MPI_STATUS_IGNORE),
      //             "Irecv");
      BN_MPI_CALL(Irecv(buffer,numItems*sizeof(T),MPI_BYTE,
                        fromRank,tag,comm,&req),
                  "Irecv");
    }
    
    template<typename T>
    inline void Comm::send(int toRank, int tag,
                           const T *buffer, int numItems, MPI_Request &req)
    {
      // BN_MPI_CALL(Send(buffer,numItems*sizeof(T),MPI_BYTE,
      //                  toRank,tag,comm),
      //             "Isend");
      BN_MPI_CALL(Isend(buffer,numItems*sizeof(T),MPI_BYTE,
                        toRank,tag,comm,&req),
                  "Isend");
    }
    
    /*! master-side of a gather where clietn gathers a fixed number
      of itmes from each rank */
    template<typename T>
    inline void Comm::masterGather(// where we'll receive into - for ALL ranks
                                   T *recvBuffer,
                                   // what we're sending (this rank's data only)
                                   const T *sendBuffer, int numItemsSentOnEachRank)
    {
      BN_MPI_CALL(Gather(sendBuffer,numItemsSentOnEachRank*sizeof(T),MPI_BYTE,
                          recvBuffer,numItemsSentOnEachRank*sizeof(T),MPI_BYTE,
                          0,comm),
                  "Gather - recv");
    }
    
    /*! client-side of a gather where each client send a fixed number
      of items to the master */
    template<typename T>
    inline void Comm::masterGather(// what we're sending (this rank's data only)
                                   const T *sendBuffer, int numItemsSentOnEachRank)
    {
      BN_MPI_CALL(Gather(sendBuffer,numItemsSentOnEachRank*sizeof(T),MPI_BYTE,
                          nullptr,numItemsSentOnEachRank*sizeof(T),MPI_BYTE,
                          0,comm),
                  "Gather - send");
    }
    

  }
}
