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

#include "barney/DeviceContext.h"
#include "barney/Ray.h"
#include "barney/Model.h"
#include "barney/fb/FrameBuffer.h"

namespace barney {
  
  DeviceContext::DeviceContext(Device::SP device)
    : device(device),
      rays(device.get())
  {}

  /* see generateRays.cu for implementation */
  __global__
  void g_generateRays(Camera camera,
                      int rngSeed,
                      vec2i fbSize,
                      int *d_count,
                      Ray *rayQueue,
                      TileDesc *tileDescs);
  
  void  DeviceContext::generateRays_sync()
  {
    SetActiveGPU forDuration(device);

    this->launch_sync();
    rays.swap();
    rays.numActive = *rays.d_nextWritePos;
    *rays.d_nextWritePos = 0;
  }
  
  void DeviceContext::shadeRays_sync()
  {
    SetActiveGPU forDuration(device);
    launch_sync();
    rays.swap();
    rays.numActive = *rays.d_nextWritePos;
    *rays.d_nextWritePos = 0;
  }

  void DeviceContext::traceRays_launch(Model *model)
  {
    DevGroup *dg = device->devGroup;
    owlParamsSetPointer(dg->lp,"rays",rays.traceAndShadeReadQueue);
    owlParamsSet1i(dg->lp,"numRays",rays.numActive);
    OWLGroup world = model->getDG(dg->ldgID)->instances.group;
    owlParamsSetGroup(dg->lp,"world",world);
    int bs = 1024;
    int nb = divRoundUp(rays.numActive,bs);
    if (nb)
      owlAsyncLaunch2DOnDevice(dg->rg,bs,nb,device->owlID,dg->lp);
  }
}
