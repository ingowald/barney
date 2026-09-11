// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


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
