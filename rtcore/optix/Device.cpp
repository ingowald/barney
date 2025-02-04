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

#include "rtcore/optix/Device.h"
#include "rtcore/optix/Denoiser.h"

namespace barney {
  namespace optix {

    Device::Device(int physicalGPU)
      : cuda::BaseDevice(physicalGPU)
    {
      owl = owlContextCreate(&physicalGPU,1);
    }

    Device::~Device()
    {
      destroy();
    }

    rtc::Denoiser *Device::createDenoiser()
    {
#if !OPTIX_DISABLE_DENOISING && OPTIX_VERSION >= 80000
      return new Denoiser(this);
#else
      // we only support optix 8 denoiser
      return nullptr;
#endif
    }

    void Device::destroy()
    {
      if (owl) owlContextDestroy(owl);
      owl = 0;
    }



    void Device::sync()
    {
      cuda::BaseDevice::sync();
      for (auto s : activeTraceStreams)
        cudaStreamSynchronize(s);
      activeTraceStreams.clear();
    }
    
    
    // ==================================================================
    // rtcore/pipeline stuff
    // ==================================================================
    void Device::buildPipeline() 
    {
      owlBuildPipeline(owl);
    }
      
    void Device::buildSBT() 
    {
      owlBuildSBT(owl);
    }
      
    

    // ==================================================================
    // buffer
    // ==================================================================

    void Device::freeBuffer(rtc::Buffer *buffer) 
    {
      delete buffer;
    }
    
    rtc::Buffer *Device::createBuffer(size_t numBytes,
                                      const void *initValues) 
    {
      return new Buffer(this,numBytes,initValues);
    }
    
    // ==================================================================
    // geomtype
    // ==================================================================
    
    rtc::GeomType *
    Device::createTrianglesGeomType(const char *ptxName,
                                    const char *typeName,
                                    size_t sizeOfDD,
                                    bool has_ah,
                                    bool has_ch)
    {
      return new TrianglesGeomType(this,ptxName,typeName,sizeOfDD,has_ah,has_ch);
    }
    
    rtc::GeomType *
    Device::createUserGeomType(const char *ptxName,
                               const char *typeName,
                               size_t sizeOfDD,
                               bool has_ah,
                               bool has_ch)
    {
      return new UserGeomType(this,ptxName,typeName,sizeOfDD,has_ah,has_ch);
    }
    
    
    // ==================================================================
    // kernels
    // ==================================================================

    struct ComputeWrapper : public ::barney::rtc::Compute {
      typedef void (*LaunchFct)(vec3i, vec3i, int,
                                cudaStream_t, const void *);
      
      ComputeWrapper(Device *device,
                     const std::string &kernelName)
        : rtc::Compute(device)
      {
        const std::string symbolName
          = "barney_rtc_cuda_launch_"+kernelName;
        launchFct = (LaunchFct)rtc::getSymbol(symbolName);
      }
      
      LaunchFct launchFct = 0;
      
      void launch(int numBlocks,
                  int blockSize,
                  const void *dd) override
      {
        do_launch(vec3i(numBlocks,1,1),vec3i(blockSize,1,1),dd);
      }
      void launch(vec2i nb,
                  vec2i bs,
                  const void *dd) override
      {
        do_launch(vec3i(nb.x,nb.y,1),vec3i(bs.x,bs.y,1),dd);
      }
      void launch(vec3i nb,
                  vec3i bs,
                  const void *dd) override
      {
        do_launch(nb,bs,dd);
      }
      void do_launch(vec3i nb,
                     vec3i bs,
                     const void *dd) 
      {
        cuda::SetActiveGPU forDuration(device);
        if (nb.x == 0) return;
        launchFct(nb,bs,0,((cuda::BaseDevice*)device)->stream,dd);

        // cudaDeviceSynchronize();
      }
    };
    
    rtc::Compute *
    Device::createCompute(const std::string &kernelName) 
    {
      return new ComputeWrapper(this,kernelName);
    }


    struct TraceWrapper : public rtc::Trace
    {
      TraceWrapper(optix::Device *device,
                   const std::string &kernelName,
                   size_t sizeOfLP)
        : Trace(device),
          device(device)
      {
        const char *ptxCode = (const char *)
          rtc::getSymbol(kernelName+"_ptx");
        OWLVarDecl rg_args[]
          = {
          { nullptr }
        };
        mod = owlModuleCreate(device->owl,ptxCode);
        rg = owlRayGenCreate(device->owl,mod,
                             kernelName.c_str(),
                             0,rg_args,-1);
        owlBuildPrograms(device->owl);

        OWLVarDecl lp_args[]
          = {
          { "raw", (OWLDataType)(OWL_USER_TYPE_BEGIN+sizeOfLP), 0 },
          { nullptr }
        };
        lp = owlParamsCreate(device->owl,sizeOfLP,lp_args,-1);
        lpStream = owlParamsGetCudaStream(lp,0);
      }

      void launch(vec2i dims, const void *dd) override
      {
        // 'in theory' we should be able to share the stream between
        // cuda kernels and trace kernels; but at least right now we
        // can't tell owl which stream to use, so owl generates its
        // own stream (per lp, to boot). so we can't currently
        // schedule this trace call into the existing cuda stream, and
        // consequently have to do a manual sync here to make sure
        // that whatever is in the cuda stream is already done by the
        // time the owl trace gets launched
        if (dims.x == 0 || dims.y == 0) return;
        
        cuda::SetActiveGPU forDuration(device);
        BARNEY_CUDA_CALL(StreamSynchronize(/*inherited!*/device->stream));
        
        owlParamsSetRaw(lp,"raw",dd,0);
        owlAsyncLaunch2D(rg,dims.x,dims.y,lp);

        device->activeTraceStreams.push_back(lpStream);
        // owlAsyncLaunch2DOnDevice(rg,dims.x,dims.y,0,lp);

        // owlLaunchSync(lp);
      }
      
      void launch(int launchDims,
                  const void *dd) override
      {
        BARNEY_NYI();
      }
      void sync() override
      {
        printf("optixbe syn lp %p\n",lp);
        cuda::SetActiveGPU forDuration(device);
        owlLaunchSync(lp);
        BARNEY_CUDA_CALL(StreamSynchronize(lpStream));
        BARNEY_CUDA_CALL(StreamSynchronize(device->stream));
        // cudaDeviceSynchronize();
      }
     
      OWLModule mod;
      OWLRayGen rg;
      OWLLaunchParams lp;
      cudaStream_t lpStream;
      optix::Device *const device;
    };
    
    rtc::Trace *
    Device::createTrace(const std::string &kernelName,
                          size_t sizeOfLP)
    {
      return new TraceWrapper(this,kernelName,sizeOfLP);
    }

    
    // ==================================================================
    // groups
    // ==================================================================
    void Device::freeGroup(rtc::Group *group)
    {
      delete group;
    }

    rtc::Group *
    Device::createTrianglesGroup(const std::vector<rtc::Geom *> &geoms)
    {
      std::vector<OWLGeom> owlGeoms;
      for (auto geom : geoms)
        owlGeoms.push_back(((Geom *)geom)->owl);
      OWLGroup g = owlTrianglesGeomGroupCreate(owl,
                                               owlGeoms.size(),
                                               owlGeoms.data());
      return new Group(this,g);
    }

    rtc::Group *
    Device::createUserGeomsGroup(const std::vector<rtc::Geom *> &geoms)
    {
      std::vector<OWLGeom> owlGeoms;
      for (auto geom : geoms)
        owlGeoms.push_back(((Geom *)geom)->owl);
      OWLGroup g = owlUserGeomGroupCreate(owl,
                                          owlGeoms.size(),
                                          owlGeoms.data());
      return new Group(this,g);
    }

    rtc::Group *
    Device::createInstanceGroup(const std::vector<rtc::Group *> &groups,
                                  const std::vector<affine3f> &xfms)
    {
      std::vector<OWLGroup> owls;
      for (auto group : groups)
        owls.push_back(((Group *)group)->owl);
      OWLGroup g
        = owlInstanceGroupCreate(owl,
                                 owls.size(),
                                 owls.data(),
                                 nullptr,
                                 (const float *)xfms.data());
      return new Group(this,g);
    }

  }
}

