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

using namespace owl::common;

int main(int, char **)
{
  int numSeconds = 20;
  
  std::cout << "creating first context" << std::endl;
  BNContext barney = bnContextCreate();
  std::cout << "destroying first context" << std::endl;
  bnContextDestroy(barney);

  std::cout << "creating/destroying contexts for " << numSeconds << " seconds" << std::endl;
  double t0 = getCurrentTime();
  int numTimesContextCreatedAndDestroyed = 1;
  while (getCurrentTime() - t0 < numSeconds) {
    barney = bnContextCreate();
    bnContextDestroy(barney);
    numTimesContextCreatedAndDestroyed++;
  }
  std::cout << "all good!" << std::endl;
  std::cout << "(note: done a total of " << prettyNumber(numTimesContextCreatedAndDestroyed)
            << " bnContextCreate()/bnContextDestry()'s)" << std::endl;
  return 0;
}

