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

#include "rtcore/cudaCommon/cuda-helper.h"
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace rtc {
  namespace optix {

    /*! Simple DeviceBuffer implementation inspired by VisRTX
        Provides safer CUDA memory management with RAII */
    struct DeviceBuffer
    {
      DeviceBuffer() = default;
      ~DeviceBuffer();

      template <typename T>
      void upload(const T *src, size_t numElements = 1, size_t byteOffsetStart = 0);

      template <typename T>
      void upload(const std::vector<T> &v);

      template <typename T>
      void download(T *dst, size_t numElements = 1, size_t byteOffsetStart = 0);

      template <typename T>
      T *ptrAs() const;
      void *ptr() const;
      size_t bytes() const;

      void reset();
      void reserve(size_t bytes);

      operator bool() const;

    private:
      template <typename T>
      size_t bytesof(size_t numElements);

      void alloc(size_t bytes);

      void *m_ptr = nullptr;
      size_t m_bytes = 0;
    };

    // Inlined definitions ////////////////////////////////////////////////////////

    inline DeviceBuffer::~DeviceBuffer() 
    {
      reset();
    }

    template <typename T>
    inline void DeviceBuffer::upload(
        const T *src, size_t numElements, size_t byteOffsetStart)
    {
      static_assert(std::is_trivially_copyable<T>::value);

      if (numElements == 0)
        return;

      auto neededBytes = bytesof<T>(numElements) + byteOffsetStart;
      if (neededBytes > bytes())
        alloc(neededBytes);

      BARNEY_CUDA_CALL(Memcpy((uint8_t *)ptr() + byteOffsetStart,
                              src,
                              bytesof<T>(numElements),
                              cudaMemcpyHostToDevice));
    }

    template <typename T>
    inline void DeviceBuffer::upload(const std::vector<T> &v)
    {
      upload(v.data(), v.size());
    }

    template <typename T>
    inline void DeviceBuffer::download(
        T *dst, size_t numElements, size_t byteOffsetStart)
    {
      static_assert(std::is_trivially_copyable<T>::value);

      if (numElements == 0)
        return;

      if (!ptr())
        throw std::runtime_error("downloading from empty DeviceBuffer");
      const auto requestedBytes = bytesof<T>(numElements);
      if ((requestedBytes + byteOffsetStart) > bytes())
        throw std::runtime_error("downloading too much data from DeviceBuffer");
      BARNEY_CUDA_CALL(Memcpy(dst,
                              (uint8_t *)ptr() + byteOffsetStart,
                              requestedBytes,
                              cudaMemcpyDeviceToHost));
    }

    template <typename T>
    inline T *DeviceBuffer::ptrAs() const
    {
      return (T *)ptr();
    }

    inline void *DeviceBuffer::ptr() const
    {
      return m_ptr;
    }

    inline size_t DeviceBuffer::bytes() const
    {
      return m_bytes;
    }

    inline void DeviceBuffer::reset()
    {
      if (m_ptr) {
        BARNEY_CUDA_CALL_NOTHROW(Free(m_ptr));
        m_ptr = nullptr;
      }
      m_bytes = 0;
    }

    inline void DeviceBuffer::reserve(size_t numBytes)
    {
      if (numBytes > bytes())
        alloc(numBytes);
    }

    inline DeviceBuffer::operator bool() const
    {
      return ptr() != nullptr;
    }

    template <typename T>
    inline size_t DeviceBuffer::bytesof(size_t numElements)
    {
      return sizeof(T) * numElements;
    }

    inline void DeviceBuffer::alloc(size_t bytes)
    {
      reset(); // Free old memory first
      BARNEY_CUDA_CALL(Malloc(&m_ptr, bytes));
      m_bytes = bytes;
    }

  } // namespace optix
} // namespace rtc
