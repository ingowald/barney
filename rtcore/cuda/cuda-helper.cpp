// ======================================================================== //
// Copyright 2019-2020 Ingo Wald                                            //
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

#include "cuda-helper.h"

#ifdef _WIN32

// // Custom usleep function for Windows
// void usleep(__int64 usec)
// {
//     // Convert microseconds to milliseconds (1 millisecond = 1000 microseconds)
//     // Minimum sleep time is 1 millisecond
//     __int64 msec = (usec / 1000 > 0) ? (usec / 1000) : 1;

//     // Use the Sleep function from Windows API
//     Sleep(static_cast<DWORD>(msec));
// }

// // Custom sleep function for Windows, emulating Unix sleep
// void sleep(unsigned int seconds)
// {
//     // Convert seconds to milliseconds and call Sleep
//     Sleep(seconds * 1000);
// }

#endif
