// ======================================================================== //
// Copyright 2023-2024 Ingo Wald                                            //
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

#include "barney.h"
#include "owl/common/math/vec.h"
#include <vector>

using namespace owl::common;

int main(int, char **)
{
  BNContext ctx = bnContextCreate();
  double t0 = getCurrentTime();

  int numSecondsToRun = 30;

  
  while (getCurrentTime() - t0 < numSecondsToRun) {
    BNFrameBuffer fb = bnFrameBufferCreate(ctx,0);

    int sx = 16 + (random() % 1024);
    int sy = 16 + (random() % 1024);\

    if (random() % 2) {
      std::vector<uint32_t> hostFB(sx*sy);
      std::vector<float>    hostDepth(sx*sy);
      bnFrameBufferResize(fb,sx,sy,hostFB.data(),hostDepth.data());
    } else {
      std::vector<uint32_t> hostFB(sx*sy);
      bnFrameBufferResize(fb,sx,sy,hostFB.data(),nullptr);
    }

    bnRelease(fb);
  }

  bnContextDestroy(ctx);
}

