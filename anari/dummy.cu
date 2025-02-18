#include "../barney/common/barney-common.h"

__global__ void dummyKernel()
{
  printf("dummuy\n");
}

extern "C" void dummy_anari()
{
  dummyKernel<<<32, 32>>>();
 
}