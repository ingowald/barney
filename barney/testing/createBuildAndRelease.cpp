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

#include "barney.h"
#include "owl/common/math/vec.h"

using namespace owl::common;


int main(int, char **)
{
  BNContext ctx = bnContextCreate();
  double t0 = getCurrentTime();

  int numSecondsToRun = 30;
  
  while (getCurrentTime() - t0 < numSecondsToRun) {
    BNModel model = bnModelCreate(ctx);
    BNDataGroup dg = bnGetDataGroup(model,0);
    // bnGroupCreate(BNDataGroup dataGroup,
    //               BNGeom *geoms, int numGeoms,
    //               BNVolume *volumes, int numVolumes);

    // bnBuild(dg);
    bnRelease(model);
  }

  bnContextDestroy(ctx);
}

