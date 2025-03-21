// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#include "Frame.h"
// std
#include <algorithm>
#include <chrono>
#include <iostream>
// cuda
#if BANARI_HAVE_CUDA
# include <cuda_runtime.h>
#endif

namespace barney_device {

  Frame::Frame(BarneyGlobalState *s) : helium::BaseFrame(s), m_renderer(this)
  {
    m_bnFrameBuffer = bnFrameBufferCreate(s->context, 0);
  }

  Frame::~Frame()
  {
    wait();
    cleanup();
    bnRelease(m_bnFrameBuffer);
  }

  bool Frame::isValid() const
  {
    return m_renderer && m_renderer->isValid() && m_camera && m_camera->isValid()
      && m_world && m_world->isValid();
  }

  BarneyGlobalState *Frame::deviceState() const
  {
    return (BarneyGlobalState *)helium::BaseObject::m_state;
  }

  BNDataType toBarney(anari::DataType type)
  {
    switch (type) {
    case ANARI_UFIXED8_VEC4:
      return BN_UFIXED8_RGBA;
    case ANARI_UFIXED8_RGBA_SRGB:
      return BN_UFIXED8_RGBA_SRGB;
    case ANARI_FLOAT32:
      return BN_FLOAT;
    case ANARI_FLOAT32_VEC3:
      return BN_FLOAT3;
    case ANARI_FLOAT32_VEC4:
      return BN_FLOAT4;
    }
    return BN_DATA_UNDEFINED;
  }

  void Frame::commitParameters()
  {
    m_renderer = getParamObject<Renderer>("renderer");
    m_camera = getParamObject<Camera>("camera");
    m_world = getParamObject<World>("world");
    m_colorType = getParam<anari::DataType>("channel.color", ANARI_UNKNOWN);
    m_depthType = getParam<anari::DataType>("channel.depth", ANARI_UNKNOWN);
    m_frameData.size = getParam<math::uint2>("size", math::uint2(10, 10));
  }

  void Frame::finalize()
  {
    cleanup();

    if (!m_renderer) {
      reportMessage(ANARI_SEVERITY_WARNING,
                    "missing required parameter 'renderer' on frame");
    }

    if (!m_camera) {
      reportMessage(ANARI_SEVERITY_WARNING, "missing required parameter 'camera' on frame");
    }

    if (!m_world) {
      reportMessage(ANARI_SEVERITY_WARNING, "missing required parameter 'world' on frame");
    }

    const auto &size = m_frameData.size;
    const auto numPixels = size.x * size.y;

    bnFrameBufferResize(m_bnFrameBuffer,
                        size.x,
                        size.y,
                        (uint32_t)BN_FB_COLOR
                        | (uint32_t)((m_depthType == ANARI_FLOAT32) ? BN_FB_DEPTH : 0));
  }

  bool Frame::getProperty(
                          const std::string_view &name, ANARIDataType type, void *ptr, uint32_t flags)
  {
    if (type == ANARI_FLOAT32 && name == "duration") {
      if (flags & ANARI_WAIT)
        wait();
      helium::writeToVoidP(ptr, m_duration);
      return true;
    }

    return 0;
  }

  void Frame::renderFrame()
  {
    auto start = std::chrono::steady_clock::now();

    auto *state = deviceState();
    state->commitBuffer.flush();

    if (m_lastCommitFlush < state->commitBuffer.lastObjectFinalization()) {
      m_lastCommitFlush = helium::newTimeStamp();
      bnAccumReset(m_bnFrameBuffer);
    }

    if (!isValid()) {
      reportMessage(
                    ANARI_SEVERITY_ERROR, "skipping render of incomplete frame object");
      reportMessage(ANARI_SEVERITY_DEBUG,
                    "    renderer(%p) - isValid:(%i)",
                    m_renderer.get(),
                    m_renderer ? m_renderer->isValid() : 0);
      reportMessage(ANARI_SEVERITY_DEBUG,
                    "    world(%p) - isValid:(%i)",
                    m_world.ptr,
                    m_world ? m_world->isValid() : 0);
      reportMessage(ANARI_SEVERITY_DEBUG,
                    "    camera(%p) - isValid:(%i)",
                    m_camera.ptr,
                    m_camera ? m_camera->isValid() : 0);
      return;
    }

    auto model = m_world->makeCurrent();

    bnRender(m_renderer->barneyRenderer,
             model,
             m_camera->barneyCamera(),
             m_bnFrameBuffer);

    auto end = std::chrono::steady_clock::now();
    m_duration = std::chrono::duration<float>(end - start).count();
  }

  void *Frame::map(std::string_view channel,
                   uint32_t *width,
                   uint32_t *height,
                   ANARIDataType *pixelType)
  {
    wait();

    *width = m_frameData.size.x;
    *height = m_frameData.size.y;
    int numPixels = *width * *height;

    if (channel == "channel.color") {
      if (m_colorBuffer)
        throw std::runtime_error
          ("trying to map color buffer, but color buffer already mapped");
      m_colorBuffer =
        new uint32_t[numPixels * (m_colorType == ANARI_FLOAT32_VEC4 ? 4 : 1)];
      bnFrameBufferRead(m_bnFrameBuffer, BN_FB_COLOR,
                        m_colorBuffer, toBarney(m_colorType));
      *pixelType = m_colorType;
      return m_colorBuffer;
    } else if (channel == "channel.depth" && m_depthBuffer) {
      if (m_depthBuffer)
        throw std::runtime_error
          ("trying to map depth buffer, but depth buffer already mapped");
      m_depthBuffer =
        new float[numPixels];
      bnFrameBufferRead(m_bnFrameBuffer, BN_FB_DEPTH, m_depthBuffer, BN_FLOAT);
      *pixelType = ANARI_FLOAT32;
      return m_depthBuffer;
    } else if (channel == "channel.colorCUDA") {
#if BANARI_HAVE_CUDA
      if (m_colorBuffer)
        throw std::runtime_error
          ("trying to map color buffer, but color buffer already mapped");
      cudaMalloc((void**)&m_colorBuffer,numPixels*
                 (m_colorType == ANARI_FLOAT32_VEC4 ? sizeof(math::float4) : sizeof(uint32_t))
                 );
      bnFrameBufferRead(m_bnFrameBuffer, BN_FB_COLOR,
                        m_colorBuffer, toBarney(m_colorType));
      *pixelType = m_colorType;
      return m_colorBuffer;
#else
      return nullptr;
#endif
    } else if (channel == "channel.depthCUDA") {
#if BANARI_HAVE_CUDA
      if (m_depthBuffer)
        throw std::runtime_error
          ("trying to map depth buffer, but depth buffer already mapped");
      cudaMalloc((void**)&m_depthBuffer,numPixels*sizeof(float));
      bnFrameBufferRead(m_bnFrameBuffer, BN_FB_DEPTH, m_depthBuffer, BN_FLOAT);
      *pixelType = ANARI_FLOAT32;
      return m_depthBuffer;
#else
      return nullptr;
#endif
    }

    *width = 0;
    *height = 0;
    *pixelType = ANARI_UNKNOWN;
    return nullptr;
  }

  void Frame::unmap(std::string_view channel)
  {
    if (channel == "channel.color") {
      if (m_colorBuffer) delete[] m_colorBuffer;
      m_colorBuffer = 0;
    } else if (channel == "channel.depth" && m_depthBuffer) {
      if (m_depthBuffer) delete[] m_depthBuffer;
      m_depthBuffer = 0;
    } else if (channel == "channel.colorCUDA") {
#if BANARI_HAVE_CUDA
      if (m_colorBuffer) cudaFree(m_colorBuffer);
      m_colorBuffer = 0;
#endif
    } else if (channel == "channel.depthCUDA") {
#if BANARI_HAVE_CUDA
      if (m_depthBuffer) cudaFree(m_depthBuffer);
      m_depthBuffer = 0;
#endif
    }
  }

  int Frame::frameReady(ANARIWaitMask m)
  {
    if (m == ANARI_NO_WAIT)
      return ready();
    else {
      wait();
      return 1;
    }
  }

  void Frame::discard()
  {
    // no-op (not yet async)
  }

  bool Frame::ready() const
  {
    return true; // not yet async
  }

  void Frame::wait() const {}

  void Frame::cleanup()
  {
    delete[] m_colorBuffer;
    delete[] m_depthBuffer;
    m_colorBuffer = nullptr;
    m_depthBuffer = nullptr;
  }

} // namespace barney_device

BARNEY_ANARI_TYPEFOR_DEFINITION(barney_device::Frame *);
