// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


/*! \file anari/dummy.cpp - Intentionally empty CUDA file whose only
    purpose is to 'force' cmake to link in CUDA runtime to the
    generated library (it is likely a cmake bug that this is even
    required, but it works for us, so this is what we do) */
#include <stdio.h>
#include <stdlib.h>

__global__ void dummyKernel()
{
  /* nothing */
}

extern "C" void dummy_anari()
{
  dummyKernel<<<32, 32>>>();
}
